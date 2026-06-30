#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>
#include <time.h>
#include <config.h>
#include <unistd.h>
#define N 256
#define DIMENSION_TILE 32

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

struct MascaraDevice {
    double **device;
    double **host_punteros;
};

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

__global__ void filtrar(const PixelU8 *entrada, PixelU8 *filtrada, double ** mascara, int tamaño_mascara, int ancho, int size, int lado_tile, int tile_size);

void resaltar(PixelU8 *data, int size, int ancho);

__global__ void umbralizar(const PixelU8 *entrada, unsigned char *salida, int size, int umbral);

__global__ void pasar_a_gris(PixelU8 *data, int size);

__global__ void calcular_sobel(const PixelU8 *data, PixelS16 *data_sobel_horizontal, PixelS16 *data_sobel_vertical, double **mascara_sobel_horizontal, double **mascara_sobel_vertical, const int tamano_mascara_sobel, const int size, const int ancho, const int lado_tile, const int tile_size);

__global__ void combinar_sobel(PixelU8 *data, const PixelS16 *data_sobel_horizontal, const PixelS16 *data_sobel_vertical, const int size);

MascaraDevice construir_mascara_sobel(enum tipo_sobel);

__device__ PixelS16 aplicar_mascara(const PixelU8 *data, int indice_pixel, int tamaño_mascara, double **mascara, double factor_normalización, int ancho, int size);

__host__ __device__ short normalizar_valor(short valor);

MascaraDevice construir_mascara_filtrado(int size);

__global__ void posterizar(PixelU8 *salida, int size, int valor_max_rgb, int rango_posterizado);

__host__ __device__ short posterizar_valor(short valor, int valor_max_rgb, int rango_posterizado);

__global__ void unir_imagenes(const unsigned char *mascara, const PixelU8 *data_posterizada, PixelU8 *resultado,
                              int size);

void guardar_imagen(const Imagen &, char *);

__host__ void construir_resultado(const Imagen &imagen_original, Imagen &resultado, const PixelU8 *data_resultado);

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

    int M_map  = (imagen_original.size + N - 1) / N;

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
    posterizar<<<M_map,N>>>(data_posterizada_device, imagen_original.size, config.valor_max_rgb, config.rango_posterizado);

    log_tiempo_etapa("[Posterizando Imagen]");
    unir_imagenes<<<M_map, N>>>(mascara_bordes_device, data_posterizada_device, data_resultado, imagen_original.size);
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

__host__ void construir_resultado(const Imagen &imagen_original, Imagen &resultado, const PixelU8 *data_resultado) {
    resultado.size = imagen_original.size;
    resultado.alto = imagen_original.alto;
    resultado.ancho = imagen_original.ancho;
    resultado.data = (PixelU8 *) asignar_memoria(resultado.size, sizeof(PixelU8));

    cudaMemcpy(resultado.data, data_resultado, resultado.size * sizeof(PixelU8), cudaMemcpyDeviceToHost);
}

__global__ void posterizar(PixelU8 *salida, const int size, const int valor_max_rgb, const int rango_posterizado) {
    const unsigned int thread_id_global = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int cantidad_threads = gridDim.x * blockDim.x;

    for (int i = thread_id_global; i < size; i += cantidad_threads) {
        salida[i].r = (unsigned char) posterizar_valor(salida[i].r, valor_max_rgb, rango_posterizado);
        salida[i].g = (unsigned char) posterizar_valor(salida[i].g, valor_max_rgb, rango_posterizado);
        salida[i].b = (unsigned char) posterizar_valor(salida[i].b, valor_max_rgb, rango_posterizado);
    }
}

void liberar_mascara_device(const MascaraDevice &m, const int size) {
    for (int i = 0; i < size; i++)
        cudaFree(m.host_punteros[i]);
    free(m.host_punteros);
    cudaFree(m.device);
}

void detectar_bordes(const PixelU8 *entrada, unsigned char *salida, const int size, const int ancho) {
    log_tiempo_etapa("[Detección de Bordes] Filtrado");


    PixelU8 *filtrada;
    cudaMalloc((void **) &filtrada, sizeof(PixelU8) * size);
    const MascaraDevice mascara = construir_mascara_filtrado(config.tamaño_mascara);
    const int radio = config.tamaño_mascara / 2;

    const int lado_tile = (DIMENSION_TILE + 2*radio);
    const int tile_size = lado_tile * lado_tile;

    int M_map  = (size + N - 1) / N;
    int M_tile = lado_tile * lado_tile;

    filtrar<<<M_tile, N, tile_size * sizeof(PixelU8)>>>(entrada, filtrada, mascara.device, config.tamaño_mascara,  ancho, size, lado_tile, tile_size);

    log_tiempo_etapa("[Detección de Bordes] Resaltado");
    resaltar(filtrada, size, ancho);
    log_tiempo_etapa("[Detección de Bordes] Umbralizado");
    umbralizar<<<M_map,N>>>(filtrada, salida, size, config.umbral);

    cudaFree(filtrada);
    liberar_mascara_device(mascara, config.tamaño_mascara);
}

__device__ void construir_tile(const PixelU8 *data, const int ancho_data, const int size_data,PixelU8 *tile, const int lado_tile, const int tile_size, const int radio, const int indice_primer_elemento) {
    for (int i = threadIdx.x; i < tile_size; i+= blockDim.x) {
        const int n_fila_tile = i / lado_tile;
        const unsigned int offset_fila = (n_fila_tile - radio) * ancho_data;
        const int n_columna_tile = i % lado_tile;
        const unsigned int offset_columna = n_columna_tile - radio;

        const int indice = indice_primer_elemento + offset_columna + offset_fila;
        const int n_fila_primer_elemento = indice_primer_elemento / ancho_data;

        const int n_fila_indice = indice / ancho_data;

        if (indice < 0 || indice >= size_data || (n_fila_indice - n_fila_primer_elemento + radio) != n_fila_tile) {
            tile[i].r = 0;
            tile[i].g = 0;
            tile[i].b = 0;

            continue;
        }

        tile[i].r = data[indice].r;
        tile[i].g = data[indice].g;
        tile[i].b = data[indice].b;
    }
}

__global__ void filtrar(const PixelU8 *entrada, PixelU8 *filtrada, double **mascara, const int tamaño_mascara, const int ancho, const int size, const int lado_tile, const int tile_size) {
    extern __shared__ PixelU8 tile_shared[];
    const int alto = size / ancho;
    const int radio = tamaño_mascara / 2;
    const int lado_tile_util = lado_tile - 2 * radio;
    const int tile_size_util = lado_tile_util * lado_tile_util;
    const int tiles_eje_x = ceil((double) ancho / lado_tile_util);
    const int tiles_eje_y = ceil((double) alto / lado_tile_util);

    const unsigned int thread_id_local = threadIdx.x;
    const unsigned int cantidad_threads_local = blockDim.x;

    for (int i = blockIdx.x; i < tiles_eje_x * tiles_eje_y; i += gridDim.x ) {
        const int coordinada_bloque_x = (i % tiles_eje_x) * lado_tile_util;
        const int coordenada_bloque_y = (i / tiles_eje_x) * lado_tile_util;

        const int indice_primer_elemento = coordenada_bloque_y * ancho + coordinada_bloque_x;

        construir_tile(entrada, ancho, size, tile_shared, lado_tile, tile_size, radio, indice_primer_elemento);

        __syncthreads();

        for (int j = thread_id_local; j < tile_size_util; j += cantidad_threads_local) {
            const int coordenada_x = (j % lado_tile_util) + radio;
            const int coordenada_y = (j / lado_tile_util) + radio;
            const int indice_pixel = coordenada_y * lado_tile + coordenada_x;

            const PixelS16 valor = aplicar_mascara(tile_shared, indice_pixel, tamaño_mascara, mascara,
                                                   1.0 / (tamaño_mascara * tamaño_mascara), lado_tile,
                                                   tile_size);

            const int x_global = coordinada_bloque_x + (j % lado_tile_util);
            const int y_global = coordenada_bloque_y + (j / lado_tile_util);

            if (x_global < ancho && y_global < alto) {
                const int idx_global = y_global * ancho + x_global;
                filtrada[idx_global].r = (unsigned char) valor.r;
                filtrada[idx_global].g = (unsigned char) valor.g;
                filtrada[idx_global].b = (unsigned char) valor.b;
            }
        }
        __syncthreads();
    }
}

MascaraDevice construir_mascara_filtrado(const int size) {
    double **mascara_host_temp = (double **) asignar_memoria(size, sizeof(double *));

    for (int i = 0; i < size; i++) {
        double *fila = (double *) asignar_memoria(size, sizeof(double));
        for (int j = 0; j < size; j++) fila[j] = 1;

        cudaMalloc((void **) &mascara_host_temp[i], sizeof(double) * size);
        cudaMemcpy(mascara_host_temp[i], fila, sizeof(double) * size, cudaMemcpyHostToDevice);
        free(fila);
    }

    double **mascara_device;
    cudaMalloc((void **) &mascara_device, sizeof(double *) * size);
    cudaMemcpy(mascara_device, mascara_host_temp, sizeof(double *) * size, cudaMemcpyHostToDevice);

    return { mascara_device, mascara_host_temp };
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

__device__ PixelS16 aplicar_mascara(const PixelU8 *data, const int indice_pixel, const int tamaño_mascara, double **mascara, const double factor_normalización, const int ancho, const int size) {
    const int medio = tamaño_mascara / 2;
    int nuevo_r = 0, nuevo_g = 0, nuevo_b = 0;

    for (int i = 0; i < tamaño_mascara; i++) {
        const int offset_fila = (i - medio) * ancho;
        for (int j = 0; j < tamaño_mascara; j++) {
            const int offset_columna = j - medio;
            const int indice_pixel_actual = indice_pixel + offset_columna + offset_fila;
            const boolean fuera = pixel_fuera_de_limite(indice_pixel_actual, indice_pixel, offset_fila, ancho, size);

            if (!fuera) {
                nuevo_r += (short) (mascara[i][j] * data[indice_pixel_actual].r);
                nuevo_g += (short) (mascara[i][j] * data[indice_pixel_actual].g);
                nuevo_b += (short) (mascara[i][j] * data[indice_pixel_actual].b);
            }
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
    int M_map  = (size + N - 1) / N;

    pasar_a_gris<<<M_map,N>>>(data, size);

    MascaraDevice mascara_sobel_horizontal = construir_mascara_sobel(HORIZONTAL);
    MascaraDevice mascara_sobel_vertical = construir_mascara_sobel(VERTICAL);

    PixelS16 *data_sobel_horizontal, *data_sobel_vertical;
    cudaMalloc((void **) &data_sobel_horizontal, size * sizeof(PixelS16));
    cudaMalloc((void **) &data_sobel_vertical, size * sizeof(PixelS16));

    const int radio = config.tamano_mascara_sobel / 2;

    const int lado_tile = (DIMENSION_TILE + 2*radio);
    const int tile_size = lado_tile * lado_tile;
    int M_tile = lado_tile * lado_tile;

    calcular_sobel<<<M_tile, N, tile_size * sizeof(PixelU8)>>>(data, data_sobel_horizontal, data_sobel_vertical, mascara_sobel_horizontal.device,
                                        mascara_sobel_vertical.device, config.tamano_mascara_sobel, size, ancho, lado_tile,
                                        tile_size);
    combinar_sobel<<<M_tile, N>>>(data, data_sobel_horizontal, data_sobel_vertical, size);

    cudaFree(data_sobel_horizontal);
    cudaFree(data_sobel_vertical);

    liberar_mascara_device(mascara_sobel_horizontal, config.tamano_mascara_sobel);
    liberar_mascara_device(mascara_sobel_vertical, config.tamano_mascara_sobel);
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

__global__ void calcular_sobel(const PixelU8 *data, PixelS16 *data_sobel_horizontal, PixelS16 *data_sobel_vertical, double **mascara_sobel_horizontal, double **mascara_sobel_vertical, const int tamano_mascara_sobel, const int size, const int ancho, const int lado_tile, const int tile_size) {
    extern __shared__ PixelU8 tile_shared[];
    const int alto = size / ancho;
    const int radio = tamano_mascara_sobel / 2;
    const int lado_tile_util = lado_tile - 2 * radio;
    const int tile_size_util = lado_tile_util * lado_tile_util;
    const int tiles_eje_x = ceil((double) ancho / lado_tile_util);
    const int tiles_eje_y = ceil((double) alto / lado_tile_util);

    const unsigned int thread_id_local = threadIdx.x;
    const unsigned int cantidad_threads_local = blockDim.x;

    for (int i = blockIdx.x; i < tiles_eje_x * tiles_eje_y; i += gridDim.x) {
        const int coordinada_bloque_x = (i % tiles_eje_x) * lado_tile_util;
        const int coordenada_bloque_y = (i / tiles_eje_x) * lado_tile_util;

        const int indice_primer_elemento = coordenada_bloque_y * ancho + coordinada_bloque_x;

        construir_tile(data, ancho, size, tile_shared, lado_tile, tile_size, radio, indice_primer_elemento);

        __syncthreads();

        for (int j = thread_id_local; j < tile_size_util; j += cantidad_threads_local) {
            const int coordenada_x = (j % lado_tile_util) + radio;
            const int coordenada_y = (j / lado_tile_util) + radio;
            const int indice_pixel = coordenada_y * lado_tile + coordenada_x;

            const PixelS16 valor_h = aplicar_mascara(tile_shared, indice_pixel, tamano_mascara_sobel,
                                                      mascara_sobel_horizontal, 1, lado_tile, tile_size);
            const PixelS16 valor_v = aplicar_mascara(tile_shared, indice_pixel, tamano_mascara_sobel,
                                                      mascara_sobel_vertical, 1, lado_tile, tile_size);

            const int x_global = coordinada_bloque_x + (j % lado_tile_util);
            const int y_global = coordenada_bloque_y + (j / lado_tile_util);

            if (x_global < ancho && y_global < alto) {
                const int idx_global = y_global * ancho + x_global;
                data_sobel_horizontal[idx_global] = valor_h;
                data_sobel_vertical[idx_global] = valor_v;
            }
        }

        __syncthreads();
    }
}

__global__ void combinar_sobel(PixelU8 *data, const PixelS16 *data_sobel_horizontal, const PixelS16 *data_sobel_vertical, const int size) {
    const unsigned int thread_id_global = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int cantidad_threads = gridDim.x * blockDim.x;

    for (int i = thread_id_global; i < size; i += cantidad_threads) {
        double gx = data_sobel_horizontal[i].r;
        double gy = data_sobel_vertical[i].r;
        short valor = normalizar_valor((short) sqrt(gx * gx + gy * gy));

        data[i].r = data[i].g = data[i].b = (unsigned char) valor;
    }
}
MascaraDevice construir_mascara_sobel(const enum tipo_sobel tipo) {
    const auto mascara = (double **) asignar_memoria(config.tamano_mascara_sobel, sizeof(double *));

    mascara[0] = (double *) asignar_memoria(config.tamano_mascara_sobel, sizeof(double));
    mascara[1] = (double *) asignar_memoria(config.tamano_mascara_sobel, sizeof(double));
    mascara[2] = (double *) asignar_memoria(config.tamano_mascara_sobel, sizeof(double));

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

    const auto mascara_host_de_punteros_device = (double **) malloc(sizeof(double *) * config.tamano_mascara_sobel);

    for (int i = 0; i < config.tamano_mascara_sobel; i++) {
        cudaMalloc((void **) &mascara_host_de_punteros_device[i], sizeof(double) * config.tamano_mascara_sobel);

        cudaMemcpy(mascara_host_de_punteros_device[i], mascara[i], sizeof(double) * config.tamano_mascara_sobel, cudaMemcpyHostToDevice);
    }

    double **mascara_device;
    cudaMalloc((void **) &mascara_device, sizeof(double *) * config.tamano_mascara_sobel);

    cudaMemcpy(mascara_device, mascara_host_de_punteros_device, sizeof(double *) * config.tamano_mascara_sobel, cudaMemcpyHostToDevice);

    for (int i = 0; i < config.tamano_mascara_sobel; i++) {
        free(mascara[i]);
    }

    free(mascara);

    return {mascara_device, mascara_host_de_punteros_device};
}

__global__ void umbralizar(const PixelU8 *entrada, unsigned char *salida, const int size, const int umbral) {
    const unsigned int thread_id_global = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int cantidad_threads = gridDim.x * blockDim.x;

    for (int i = thread_id_global; i < size; i += cantidad_threads) {
        salida[i] = entrada[i].r < umbral ? 0 : 1;
    }
}

__host__ __device__ short posterizar_valor(const short valor, const int valor_max_rgb, const int rango_posterizado) {
    const short ancho_rango = valor_max_rgb / rango_posterizado;
    const short offset_representativo = ancho_rango / 2;
    short rango = valor / ancho_rango;
    rango = rango >= rango_posterizado ? rango_posterizado - 1 : rango;

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
