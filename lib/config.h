#ifndef CONFIG_H
#define CONFIG_H

typedef struct {
    int desired_channels;
    int rango_posterizado;
    int valor_max_rgb;
    int tamaño_mascara;
    int tamaño_mascara_sobel;
    int umbral;
    char path_imagen[50];
} Config;

Config leer_config(const char *path);

#endif
