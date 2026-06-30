#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"

#define MAX_LINE 256
#define CONFIG_PATH "cartoon.config"

typedef struct {
    const char *key;
    int *campo;
    int encontrado;
} ConfigEntry;

Config leer_config(const char *path) {
    Config config = {0};

    ConfigEntry entries[] = {
        {"DESIRED_CHANNELS",     &config.desired_channels,    0},
        {"RANGO_POSTERIZADO",    &config.rango_posterizado,   0},
        {"VALOR_MAX_RGB",        &config.valor_max_rgb,       0},
        {"TAMAÑO_MASCARA",       &config.tamaño_mascara,      0},
        {"TAMAÑO_MASCARA_SOBEL", &config.tamano_mascara_sobel, 0},
        {"UMBRAL",               &config.umbral,              0},
    };
    const int n_entries = sizeof(entries) / sizeof(entries[0]);

    int path_hallado = 0;

    FILE *f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "ERROR:\tNo se pudo abrir el archivo de configuración: %s\nSaliendo...\n", path);
        exit(-1);
    }

    char linea[MAX_LINE];
    while (fgets(linea, sizeof(linea), f)) {
        if (linea[0] == '\n' || linea[0] == '#') continue;

        linea[strcspn(linea, "\n")] = '\0';

        char *separador = strchr(linea, '=');
        if (separador == NULL) continue;

        *separador = '\0';
        const char *key = linea;
        const char *value = separador + 1;

        if (strcmp(key, "PATH_IMAGEN") == 0) {
            strncpy(config.path_imagen, value, 49);
            path_hallado = 1;
            continue;
        }

        for (int i = 0; i < n_entries; i++) {
            if (strcmp(key, entries[i].key) == 0) {
                *entries[i].campo = atoi(value);
                entries[i].encontrado = 1;
                break;
            }
        }
    }

    fclose(f);

    for (int i = 0; i < n_entries; i++) {
        if (!entries[i].encontrado) {
            fprintf(stderr, "ERROR:\tFalta la clave '%s' en el archivo de configuración\nSaliendo...\n",
                    entries[i].key);
            exit(-1);
        }
    }

    if (!path_hallado) {
        fprintf(stderr, "ERROR:\tFalta la clave 'PATH_IMAGEN' en el archivo de configuración\nSaliendo...\n");
        exit(-1);
    }

    return config;
}
