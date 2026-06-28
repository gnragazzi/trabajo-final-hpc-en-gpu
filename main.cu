#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define M 8
#define N 128
#include <stb_image.h>
#include <stb_image_write.h>
#include <time.h>
#include <config.h>
#include <unistd.h>

typedef struct {
    unsigned char r, g, b;
} PixelU8;

typedef struct {
    short r, g, b;
} PixelS16;

typedef struct {
    PixelU8 *data;
    int alto;
    int ancho;
    int size;
} Imagen;

struct timespec inicio_global, marca;

struct timespec inicio_etapa;

enum boolean { FALSE = 0, TRUE };

enum tipo_sobel { HORIZONTAL = 0, VERTICAL };

Config config;

void log_tiempo(const char *etapa);

void iniciar_etapa();

void log_tiempo_etapa(const char *etapa);

void validar_input(int, char **);

void error(const char *msj);

void *asignar_memoria(int, size_t);

Imagen leer_imagen(const char *);

void detectar_bordes(const PixelU8 *entrada, unsigned char *salida, int size, int ancho);

void filtrar(const PixelU8 *entrada, PixelU8 *filtrada, double ** mascara, int ancho, int size);

void resaltar(PixelU8 *data, int size, int ancho);

__global__ void umbralizar(PixelU8 *entrada, unsigned char *salida, int size);

__global__ void pasar_a_gris(PixelU8 *data, int size);

__global__ void aplicar_operador_gradiente(PixelU8 *data, PixelS16 *data_sobel_horizontal, PixelS16 *data_sobel_vertical, double **mascara_sobel_horizontal, double **mascara_sobel_vertical, int size, int ancho);

double **construir_mascara_sobel(enum tipo_sobel);

PixelS16 aplicar_mascara(const PixelU8 *data, int indice_pixel, int tamaño_máscara, double **mascara, double factor_normalización, int ancho, int size);

__host__ __device__ short normalizar_valor(short valor);

__host__ __device__ double **construir_mascara_filtrado(int size);

__global__ void posterizar(PixelU8 *salida, int size);

__host__ __device__ short posterizar_valor(short valor);

__global__ void unir_imagenes(const unsigned char *mascara, const PixelU8 *data_posterizada, PixelU8 *resultado,
                              int size);

void guardar_imagen(const Imagen &, char *);

__host__ void construir_resultado(Imagen imagen_original, Imagen &resultado, PixelU8 *data_resultado);

int main(const int argc, char **argv) {
    clock_gettime(CLOCK_MONOTONIC, &inicio_global);

    char buffer[256];
    snprintf(buffer, sizeof(buffer), "Iniciando Ejecución: Modo Secuencial - Configuración: %s", argv[1]);

    log_tiempo(buffer);

    validar_input(argc, argv);

    config = leer_config(argv[1]);

    iniciar_etapa();
    const Imagen imagen_original = leer_imagen(config.path_imagen);
    Imagen resultado;
    PixelU8 *data_posterizada_device, *data_detectar_bordes_device, *data_resultado;
    unsigned char *mascara_bordes_device;

    cudaMalloc((void **) &data_posterizada_device, sizeof(PixelU8) * imagen_original.size);
    cudaMalloc((void **) &data_detectar_bordes_device, sizeof(PixelU8) * imagen_original.size);
    cudaMalloc((void **) &data_resultado, sizeof(PixelU8) * imagen_original.size);
    cudaMalloc((void **) &mascara_bordes_device, sizeof(unsigned char) * imagen_original.size);

    cudaMemcpy(data_posterizada_device, imagen_original.data, sizeof(PixelU8) * imagen_original.size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(data_detectar_bordes_device, imagen_original.data, sizeof(PixelU8) * imagen_original.size,
               cudaMemcpyHostToDevice);

    log_tiempo_etapa("[Leyendo Imagen]");

    detectar_bordes(data_detectar_bordes_device, mascara_bordes_device, imagen_original.size, imagen_original.ancho);

    iniciar_etapa();
    posterizar<<<M,N>>>(data_posterizada_device, imagen_original.size);

    log_tiempo_etapa("[Posterizando Imagen]");
    unir_imagenes<<<M, N>>>(mascara_bordes_device, data_posterizada_device, data_resultado, imagen_original.size);
    log_tiempo_etapa("[Uniendo Imágenes]");

    iniciar_etapa();

    construir_resultado(imagen_original, resultado, data_resultado);
    guardar_imagen(resultado, config.path_imagen);

    free(imagen_original.data);
    free(resultado.data);
    cudaFree(data_posterizada_device);
    cudaFree(data_detectar_bordes_device);
    cudaFree(mascara_bordes_device);
    cudaFree(data_resultado);
    log_tiempo_etapa("[Guardando Imagen]");

    snprintf(buffer, sizeof(buffer), "Finalizando Ejecución. Imagen medida X: %d Y: %d", imagen_original.ancho,
             imagen_original.alto);

    log_tiempo(buffer);

    return 0;
}

void iniciar_etapa() {
    clock_gettime(CLOCK_MONOTONIC, &inicio_etapa);
}

void log_tiempo_etapa(const char *etapa) {
    struct timespec fin;
    clock_gettime(CLOCK_MONOTONIC, &fin);

    const double elapsed = (fin.tv_sec - inicio_etapa.tv_sec) +
                           (fin.tv_nsec - inicio_etapa.tv_nsec) / 1e9;

    printf("[TIEMPO ETAPA] %s: %.6f s\n", etapa, elapsed);
}

void log_tiempo(const char *etapa) {
    clock_gettime(CLOCK_MONOTONIC, &marca);
    double elapsed = (marca.tv_sec - inicio_global.tv_sec) + (marca.tv_nsec - inicio_global.tv_nsec) / 1e9;
    printf("[%.6f s] %s\n", elapsed, etapa);
}

void validar_input(int c, char **v) {
    if (c != 2)
        error("Este programa corre únicamente con 1 argumentos: el archivo de configuración");
}

void error(char const *msj) {
    fprintf(stderr, "ERROR:\t%s\nSaliendo...\n", msj);
    exit(-1);
}

void *asignar_memoria(int n, size_t size) {
    void *temp = calloc(n, size);
    if (temp == nullptr) {
        error("Hubo un problema en la asignación de memoria");
    }
    return temp;
}

Imagen leer_imagen(const char *path) {
    Imagen imagen;
    int channels_in_file;
    char cwd[256];

    if (getcwd(cwd, sizeof(cwd)) == NULL) {
        perror("getcwd");
        exit(EXIT_FAILURE);
    }

    char image_complete_path[512];
    snprintf(image_complete_path, sizeof(image_complete_path), "%s%s", cwd, path);

    const unsigned char *file = stbi_load(image_complete_path, &imagen.ancho, &imagen.alto, &channels_in_file,
                                          config.desired_channels);

    imagen.size = imagen.alto * imagen.ancho;

    PixelU8 *pixeles = (PixelU8 *) asignar_memoria(imagen.size, sizeof(PixelU8));
    imagen.data = pixeles;

    if (file == NULL)
        error("No se pudo leer el archivo");

    if (config.desired_channels != channels_in_file)
        error("Hubo un problema al procesar la imagen");

    for (int i = 0; i < imagen.size; i++) {
        imagen.data[i].r = file[i * channels_in_file];
        imagen.data[i].g = file[i * channels_in_file + 1];
        imagen.data[i].b = file[i * channels_in_file + 2];
    }

    stbi_image_free((void *) file);

    return imagen;
}

void guardar_imagen(const Imagen &imagen, char *path) {
    const int tamaño_data_cruda = imagen.size * config.desired_channels;
    unsigned char *data = (unsigned char *) asignar_memoria(tamaño_data_cruda, sizeof(unsigned char));

    char cwd[256];
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
        perror("getcwd");
        exit(EXIT_FAILURE);
    }

    char image_complete_path[512];
    snprintf(image_complete_path, sizeof(image_complete_path), "%s%s_mask%d_umbral%d.png",
             cwd, path, config.tamaño_mascara, config.umbral);

    for (int i = 0; i < imagen.size; i++) {
        data[i * config.desired_channels] = imagen.data[i].r;
        data[i * config.desired_channels + 1] = imagen.data[i].g;
        data[i * config.desired_channels + 2] = imagen.data[i].b;
    }

    stbi_write_png(image_complete_path, imagen.ancho, imagen.alto, config.desired_channels, data, 0);

    free((void *) data);
}

__host__ void construir_resultado(const Imagen imagen_original, Imagen &resultado, PixelU8 *data_resultado) {
    resultado.size = imagen_original.size;
    resultado.alto = imagen_original.alto;
    resultado.ancho = imagen_original.ancho;
    resultado.data = (PixelU8 *) asignar_memoria(resultado.size, sizeof(PixelU8));

    cudaMemcpy(resultado.data, data_resultado, resultado.size, cudaMemcpyDeviceToHost);
}

__global__ void posterizar(PixelU8 *salida, const int size) {
    const unsigned int thread_id_global = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int cantidad_threads = gridDim.x * blockDim.x;

    for (int i = thread_id_global; i < size; i += cantidad_threads) {
        salida[i].r = (unsigned char) posterizar_valor(salida[i].r);
        salida[i].g = (unsigned char) posterizar_valor(salida[i].g);
        salida[i].b = (unsigned char) posterizar_valor(salida[i].b);
    }
}

void detectar_bordes(PixelU8 *entrada, unsigned char *salida, const int size, const int ancho) {
    log_tiempo_etapa("[Detección de Bordes] Filtrado");

    PixelU8 *filtrada;
    cudaMalloc((void **) &filtrada, sizeof(PixelU8) * size);
    double **máscara = construir_mascara_filtrado(config.tamaño_mascara);

    filtrar(entrada, filtrada, máscara, ancho, size);
    log_tiempo_etapa("[Detección de Bordes] Resaltado");
    resaltar(filtrada, size, ancho);
    log_tiempo_etapa("[Detección de Bordes] Umbralizado");
    umbralizar<<<M,N>>>(filtrada, salida, size);

    cudaFree(filtrada);
    for (int i = 0; i < config.tamaño_mascara; i++)
        cudaFree(máscara[i]);
    cudaFree(máscara);
}

__global__ void filtrar(const PixelU8 *entrada, PixelU8 *filtrada, double ** mascara, const int ancho, const int size) {
    for (int i = 0; i < size; i++) {
        const PixelS16 valor = aplicar_mascara(entrada, i, config.tamaño_mascara, mascara, 1.0 / (config.tamaño_mascara * config.tamaño_mascara), ancho, size);
        filtrada[i].r = (unsigned char) valor.r;
        filtrada[i].g = (unsigned char) valor.g;
        filtrada[i].b = (unsigned char) valor.b;
    }
}

__host__ __device__ double **construir_mascara_filtrado(const int size) {
    const auto mascara = (double **) asignar_memoria(size, sizeof(double *));

    for (int i = 0; i < size; i++) {
        mascara[i] = (double *) asignar_memoria(size, sizeof(double));
        for (int j = 0; j < size; j++) {
            mascara[i][j] = 1;
        }
    }

    double **mascara_device;
    cudaMalloc((void **) &mascara_device, sizeof(double *) * size);

    for (int i = 0; i < size; i++) {
        cudaMalloc((void **) &mascara_device[i], sizeof(double) * size);
        cudaMemcpy(mascara_device[i], mascara[i], sizeof(double) * size, cudaMemcpyHostToDevice);
    }

    for (int i = 0; i < config.tamaño_mascara; i++)
        free(mascara[i]);
    free(mascara);

    return mascara_device;
}

__host__ __device__ boolean pixel_fuera_de_limite(const int indice_pixel_actual, const int indice_pixel,
                                                  const int offset_fila,
                                                  const int ancho, const int size) {
    if (indice_pixel_actual < 0)
        return TRUE;
    if (indice_pixel_actual > (size - 1))
        return TRUE;

    const int fila_pixel = (indice_pixel + offset_fila) / ancho;
    const int fila_pixel_actual = indice_pixel_actual / ancho;
    if (fila_pixel != fila_pixel_actual)
        return TRUE;

    return FALSE;
}

__device__ PixelS16 aplicar_mascara(const PixelU8 *data, const int indice_pixel, const int tamaño_máscara, double **mascara, const double factor_normalización, const int ancho, const int size) {
    const int medio = tamaño_máscara / 2;
    PixelS16 pixeles_enmascarados[tamaño_máscara][tamaño_máscara];

    for (int i = 0; i < tamaño_máscara; i++) {
        const int offset_fila = (i - medio) * ancho;
        for (int j = 0; j < tamaño_máscara; j++) {
            const int offset_columna = j - medio;
            const int indice_pixel_actual = indice_pixel + offset_columna + offset_fila;
            const boolean fuera = pixel_fuera_de_limite(indice_pixel_actual, indice_pixel, offset_fila, ancho, size);

            pixeles_enmascarados[i][j].r = fuera ? 0 : (short) (mascara[i][j] * data[indice_pixel_actual].r);
            pixeles_enmascarados[i][j].g = fuera ? 0 : (short) (mascara[i][j] * data[indice_pixel_actual].g);
            pixeles_enmascarados[i][j].b = fuera ? 0 : (short) (mascara[i][j] * data[indice_pixel_actual].b);
        }
    }

    int nuevo_r = 0;
    int nuevo_g = 0;
    int nuevo_b = 0;

    for (int i = 0; i < tamaño_máscara; i++) {
        for (int j = 0; j < tamaño_máscara; j++) {
            nuevo_r += pixeles_enmascarados[i][j].r;
            nuevo_g += pixeles_enmascarados[i][j].g;
            nuevo_b += pixeles_enmascarados[i][j].b;
        }
    }

    PixelS16 respuesta;
    respuesta.r = (short) (nuevo_r * factor_normalización);
    respuesta.g = (short) (nuevo_g * factor_normalización);
    respuesta.b = (short) (nuevo_b * factor_normalización);

    return respuesta;
}

__host__ __device__ short normalizar_valor(const short valor) {
    if (valor < 0)
        return 0;
    if (valor > 255)
        return 255;

    return valor;
}

void resaltar(PixelU8 *data, const int size, const int ancho) {
    pasar_a_gris<<<M,N>>>(data, size);

    double **mascara_sobel_horizontal = construir_mascara_sobel(HORIZONTAL);
    double **mascara_sobel_vertical = construir_mascara_sobel(VERTICAL);

    PixelS16 *data_sobel_horizontal, *data_sobel_vertical;
    cudaMalloc((void **) &data_sobel_horizontal, size * sizeof(PixelS16));
    cudaMalloc((void **) &data_sobel_vertical, size * sizeof(PixelS16));

    aplicar_operador_gradiente<<<M,N>>>(data, data_sobel_horizontal, data_sobel_vertical, mascara_sobel_horizontal, mascara_sobel_vertical, size, ancho);

    cudaFree(data_sobel_horizontal);
    cudaFree(data_sobel_vertical);

    for (int i = 0; i < config.tamaño_mascara_sobel; i++) {
        cudaFree(mascara_sobel_horizontal[i]);
        cudaFree(mascara_sobel_vertical[i]);
    }

    cudaFree(mascara_sobel_horizontal);
    cudaFree(mascara_sobel_vertical);
}

__global__ void pasar_a_gris(PixelU8 *data, const int size) {
    const unsigned int thread_id_global = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int cantidad_threads = gridDim.x * blockDim.x;

    for (int i = thread_id_global; i < size; i += cantidad_threads) {
        const unsigned char nuevo_valor = (unsigned char) (0.3 * data[i].r + 0.59 * data[i].g + 0.11 * data[i].b);
        data[i].r = nuevo_valor;
        data[i].g = nuevo_valor;
        data[i].b = nuevo_valor;
    }
}

__global__ void aplicar_operador_gradiente(PixelU8 *data, PixelS16 *data_sobel_horizontal, PixelS16 *data_sobel_vertical, double **mascara_sobel_horizontal, double **mascara_sobel_vertical, const int size, const int ancho) {
    const unsigned int thread_id_global = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int cantidad_threads = gridDim.x * blockDim.x;


    for (int i = 0; i < size; i++) {
        data_sobel_horizontal[i] = aplicar_mascara(data, i, config.tamaño_mascara_sobel, mascara_sobel_horizontal, 1, ancho, size);
        data_sobel_vertical[i] = aplicar_mascara(data, i, config.tamaño_mascara_sobel, mascara_sobel_vertical, 1, ancho, size);
    }

    for (int i = thread_id_global; i < size; i += cantidad_threads) {
        double gx = data_sobel_horizontal[i].r;
        double gy = data_sobel_vertical[i].r;
        short valor = normalizar_valor((short) sqrt(gx * gx + gy * gy));

        data[i].r = data[i].g = data[i].b = (unsigned char) valor;
    }
}

double **construir_mascara_sobel(enum tipo_sobel tipo) {
    const auto mascara = (double **) asignar_memoria(config.tamaño_mascara_sobel, sizeof(double *));

    mascara[0] = (double *) asignar_memoria(config.tamaño_mascara_sobel, sizeof(double));
    mascara[1] = (double *) asignar_memoria(config.tamaño_mascara_sobel, sizeof(double));
    mascara[2] = (double *) asignar_memoria(config.tamaño_mascara_sobel, sizeof(double));

    if (tipo == HORIZONTAL) {
        mascara[0][0] = -1;
        mascara[0][1] = 0;
        mascara[0][2] = 1;
        mascara[1][0] = -2;
        mascara[1][1] = 0;
        mascara[1][2] = 2;
        mascara[2][0] = -1;
        mascara[2][1] = 0;
        mascara[2][2] = 1;
    } else {
        mascara[0][0] = 1;
        mascara[0][1] = 2;
        mascara[0][2] = 1;
        mascara[1][0] = 0;
        mascara[1][1] = 0;
        mascara[1][2] = 0;
        mascara[2][0] = -1;
        mascara[2][1] = -2;
        mascara[2][2] = -1;
    }

    double **mascara_device;
    cudaMalloc((void **) &mascara_device, sizeof(double *) * config.tamaño_mascara_sobel);

    cudaMalloc((void **) &mascara_device[0], sizeof(double) * config.tamaño_mascara_sobel);
    cudaMalloc((void **) &mascara_device[1], sizeof(double) * config.tamaño_mascara_sobel);
    cudaMalloc((void **) &mascara_device[2], sizeof(double) * config.tamaño_mascara_sobel);

    cudaMemcpy(mascara_device[0], mascara[0], sizeof(double) * config.tamaño_mascara_sobel, cudaMemcpyHostToDevice);
    cudaMemcpy(mascara_device[1], mascara[1], sizeof(double) * config.tamaño_mascara_sobel, cudaMemcpyHostToDevice);
    cudaMemcpy(mascara_device[2], mascara[2], sizeof(double) * config.tamaño_mascara_sobel, cudaMemcpyHostToDevice);

    for (int i = 0; i < config.tamaño_mascara_sobel; i++) {
        free(mascara[i]);
    }

    free(mascara);

    return mascara;
}

__global__ void umbralizar(PixelU8 *entrada, unsigned char *salida, int size) {
    const unsigned int thread_id_global = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int cantidad_threads = gridDim.x * blockDim.x;

    for (int i = thread_id_global; i < size; i += cantidad_threads) {
        salida[i] = entrada[i].r < config.umbral ? 0 : 1;
    }
}

__host__ __device__ short posterizar_valor(const short valor) {
    const short ancho_rango = config.valor_max_rgb / config.rango_posterizado;
    const short offset_representativo = ancho_rango / 2;
    short rango = valor / ancho_rango;
    rango = rango >= config.rango_posterizado ? config.rango_posterizado - 1 : rango;

    return rango * ancho_rango + offset_representativo;
}

__global__ void unir_imagenes(const unsigned char *mascara, const PixelU8 *data_posterizada, PixelU8 *resultado,
                              const int size) {
    const unsigned int thread_id_global = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int cantidad_threads = gridDim.x * blockDim.x;

    for (int i = thread_id_global; i < size; i += cantidad_threads) {
        if (mascara[i]) {
            resultado[i].r = 0;
            resultado[i].g = 0;
            resultado[i].b = 0;
        } else {
            resultado[i] = data_posterizada[i];
        }
    }
}
