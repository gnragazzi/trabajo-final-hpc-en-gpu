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

PixelU8 *copiar_data(const PixelU8 *, int);

Imagen leer_imagen(const char *);

unsigned char *detectar_bordes(const PixelU8 *entrada, unsigned char *salida, int size);

void filtrar(Imagen *);

void resaltar(const Imagen *);

__global__ void umbralizar(PixelU8 *entrada, unsigned char *salida, int size);

void pasar_a_gris(const Imagen *);

void aplicar_operador_gradiente(const Imagen *);

double **construir_mascara_sobel(enum tipo_sobel);

PixelS16 aplicar_mascara(const Imagen *imagen, int indice_pixel, int tamaño_máscara, double **mascara,
                         double factor_normalización);

short normalizar_valor(short valor);

double **construir_mascara_filtrado(int size);

__global__ void posterizar(PixelU8 *entrada, PixelU8 *salida, int size);

short posterizar_valor(short valor);

__global__ void unir_imagenes(const unsigned char *mascara, const PixelU8 *data_posterizada, PixelU8 *resultado,
                              int size);

void guardar_imagen(const Imagen &, char *);

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
    PixelU8 *data_imagen_original_device, *data_posterizada_device, *data_detectar_bordes_device, *data_resultado;
    unsigned char *mascara_bordes_device;

    cudaMalloc((void **) &data_imagen_original_device, sizeof(PixelU8) * imagen_original.size);
    cudaMalloc((void **) &data_posterizada_device, sizeof(PixelU8) * imagen_original.size);
    cudaMalloc((void **) &data_detectar_bordes_device, sizeof(PixelU8) * imagen_original.size);
    cudaMalloc((void **) &data_resultado, sizeof(PixelU8) * imagen_original.size);
    cudaMalloc((void **) &mascara_bordes_device, sizeof(unsigned char) * imagen_original.size);

    cudaMemcpy(data_imagen_original_device, imagen_original.data, sizeof(PixelU8) * imagen_original.size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(data_detectar_bordes_device, imagen_original.data, sizeof(PixelU8) * imagen_original.size,
               cudaMemcpyHostToDevice);

    log_tiempo_etapa("[Leyendo Imagen]");

    detectar_bordes(data_detectar_bordes_device, mascara_bordes_device, imagen_original.size);

    iniciar_etapa();
    posterizar<<<M,N>>>(data_imagen_original_device, data_posterizada_device, imagen_original.size);

    log_tiempo_etapa("[Posterizando Imagen]");
    unir_imagenes<<<M, N>>>(mascara_bordes_device, data_posterizada_device, data_resultado, imagen_original.size);
    log_tiempo_etapa("[Uniendo Imágenes]");

    iniciar_etapa();

    resultado.size = imagen_original.size;
    resultado.alto = imagen_original.alto;
    resultado.ancho = imagen_original.ancho;
    resultado.data = (PixelU8 *) asignar_memoria(resultado.size, sizeof(PixelU8));

    cudaMemcpy(resultado.data, data_resultado, resultado.size, cudaMemcpyDeviceToHost);
    guardar_imagen(resultado, config.path_imagen);

    free(imagen_original.data);
    free(resultado.data);
    cudaFree(data_imagen_original_device);
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

PixelU8 *copiar_data(const PixelU8 *data, const int size) {
    const auto copia = (PixelU8 *) asignar_memoria(size, sizeof(PixelU8));

    for (int i = 0; i < size; i++) {
        copia[i].r = data[i].r;
        copia[i].g = data[i].g;
        copia[i].b = data[i].b;
    }
    return copia;
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

__global__ void posterizar(const PixelU8 *entrada, PixelU8 *salida, const int size) {
    const unsigned int thread_id_global = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int cantidad_threads = gridDim.x * blockDim.x;

    for (int i = thread_id_global; i < size; i += cantidad_threads) {
        salida[i].r = (unsigned char) posterizar_valor(entrada[i].r);
        salida[i].g = (unsigned char) posterizar_valor(entrada[i].g);
        salida[i].b = (unsigned char) posterizar_valor(entrada[i].b);
    }
}

unsigned char *detectar_bordes(PixelU8 *entrada, unsigned char *salida, int size) {
    log_tiempo_etapa("[Detección de Bordes] Filtrado");
    filtrar(entrada);
    log_tiempo_etapa("[Detección de Bordes] Resaltado");
    resaltar(entrada);
    log_tiempo_etapa("[Detección de Bordes] Umbralizado");
    umbralizar<<<M,N>>>(entrada, salida, size);
}

void filtrar(Imagen *imagen) {
    const auto data_blureada = (PixelU8 *) asignar_memoria(imagen->size, sizeof(PixelU8));
    double **máscara = construir_mascara_filtrado(config.tamaño_mascara);

    for (int i = 0; i < imagen->size; i++) {
        PixelS16 valor = aplicar_mascara(imagen, i, config.tamaño_mascara, máscara,
                                         1.0 / (config.tamaño_mascara * config.tamaño_mascara));
        data_blureada[i].r = (unsigned char) valor.r;
        data_blureada[i].g = (unsigned char) valor.g;
        data_blureada[i].b = (unsigned char) valor.b;
    }

    free(imagen->data);
    for (int i = 0; i < config.tamaño_mascara; i++)
        free(máscara[i]);
    free(máscara);

    imagen->data = data_blureada;
}

double **construir_mascara_filtrado(const int size) {
    const auto mascara = (double **) asignar_memoria(size, sizeof(double *));

    for (int i = 0; i < size; i++) {
        mascara[i] = (double *) asignar_memoria(size, sizeof(double));
        for (int j = 0; j < size; j++) {
            mascara[i][j] = 1;
        }
    }

    return mascara;
}

enum boolean pixel_fuera_de_limite(const int indice_pixel_actual, const int indice_pixel, const int offset_fila,
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

PixelS16 aplicar_mascara(const Imagen *imagen, const int indice_pixel, const int tamaño_máscara, double **mascara,
                         double factor_normalización) {
    const int medio = tamaño_máscara / 2;
    PixelS16 pixeles_enmascarados[tamaño_máscara][tamaño_máscara];

    for (int i = 0; i < tamaño_máscara; i++) {
        const int offset_fila = (i - medio) * imagen->ancho;
        for (int j = 0; j < tamaño_máscara; j++) {
            const int offset_columna = j - medio;
            const int indice_pixel_actual = indice_pixel + offset_columna + offset_fila;
            const enum boolean fuera = pixel_fuera_de_limite(indice_pixel_actual, indice_pixel, offset_fila,
                                                             imagen->ancho, imagen->size);

            pixeles_enmascarados[i][j].r = fuera ? 0 : (short) (mascara[i][j] * imagen->data[indice_pixel_actual].r);
            pixeles_enmascarados[i][j].g = fuera ? 0 : (short) (mascara[i][j] * imagen->data[indice_pixel_actual].g);
            pixeles_enmascarados[i][j].b = fuera ? 0 : (short) (mascara[i][j] * imagen->data[indice_pixel_actual].b);
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

short normalizar_valor(const short valor) {
    if (valor < 0)
        return 0;
    if (valor > 255)
        return 255;

    return valor;
}

void resaltar(const Imagen *imagen) {
    pasar_a_gris(imagen);
    aplicar_operador_gradiente(imagen);
}

void pasar_a_gris(const Imagen *imagen) {
    for (int i = 0; i < imagen->size; i++) {
        const unsigned char nuevo_valor =
                (unsigned char) (0.3 * imagen->data[i].r + 0.59 * imagen->data[i].g + 0.11 * imagen->data[i].b);
        imagen->data[i].r = nuevo_valor;
        imagen->data[i].g = nuevo_valor;
        imagen->data[i].b = nuevo_valor;
    }
}

void aplicar_operador_gradiente(const Imagen *imagen) {
    PixelS16 *data_sobel_horizontal = (PixelS16 *) asignar_memoria(imagen->size, sizeof(PixelS16));
    PixelS16 *data_sobel_vertical = (PixelS16 *) asignar_memoria(imagen->size, sizeof(PixelS16));
    double **mascara_sobel_horizontal = construir_mascara_sobel(HORIZONTAL);
    double **mascara_sobel_vertical = construir_mascara_sobel(VERTICAL);

    for (int i = 0; i < imagen->size; i++) {
        data_sobel_horizontal[i] = aplicar_mascara(imagen, i, config.tamaño_mascara_sobel, mascara_sobel_horizontal, 1);
        data_sobel_vertical[i] = aplicar_mascara(imagen, i, config.tamaño_mascara_sobel, mascara_sobel_vertical, 1);
    }

    for (int i = 0; i < imagen->size; i++) {
        double gx = data_sobel_horizontal[i].r;
        double gy = data_sobel_vertical[i].r;
        short valor = normalizar_valor((short) sqrt(gx * gx + gy * gy));

        imagen->data[i].r = imagen->data[i].g = imagen->data[i].b = (unsigned char) valor;
    }

    free(data_sobel_horizontal);
    free(data_sobel_vertical);

    for (int i = 0; i < config.tamaño_mascara_sobel; i++) {
        free(mascara_sobel_horizontal[i]);
        free(mascara_sobel_vertical[i]);
    }

    free(mascara_sobel_horizontal);
    free(mascara_sobel_vertical);
}

double **construir_mascara_sobel(enum tipo_sobel tipo) {
    double **mascara = (double **) asignar_memoria(config.tamaño_mascara_sobel, sizeof(double *));

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
