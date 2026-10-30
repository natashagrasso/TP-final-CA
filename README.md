# Trabajo Práctico Final – Computación Avanzada

Este proyecto implementa una pipeline de generación de imágenes estilo cartoon con cuatro variantes:

- Secuencial
- OpenMP
- MPI
- Híbrida MPI + OpenMP

La imagen de entrada se convierte a una representación RGB, se aplica una etapa de detección de bordes y una etapa de posterización, y luego se combinan para obtener la imagen final con apariencia de dibujo animado.

## Estructura

- `src/bmp.h` y `src/bmp.cpp`: lectura y escritura de imágenes BMP.
- `src/cartoon_common.h`: utilidades compartidas para la detección de bordes y posterización.
- `src/cartoon_sequential.cpp`: versión secuencial.
- `src/cartoon_openmp.cpp`: versión con OpenMP.
- `src/cartoon_mpi.cpp`: versión con MPI.
- `src/cartoon_hybrid.cpp`: versión híbrida MPI + OpenMP.
- `src/generate_test_image.cpp`: generador de una imagen BMP de prueba.

## Compilación

### Secuencial

g++ -O3 -std=c++17 src/bmp.cpp src/cartoon_common.cpp src/cartoon_sequential.cpp -o cartoon_sequential.exe

### OpenMP

g++ -O3 -std=c++17 -fopenmp src/bmp.cpp src/cartoon_common.cpp src/cartoon_openmp.cpp -o cartoon_openmp.exe

### MPI

C:\msys64\ucrt64\bin\mpicxx.exe -O3 -std=c++17 src/bmp.cpp src/cartoon_common.cpp src/cartoon_mpi.cpp -o cartoon_mpi.exe

### Híbrida

C:\msys64\ucrt64\bin\mpicxx.exe -O3 -std=c++17 -fopenmp src/bmp.cpp src/cartoon_common.cpp src/cartoon_hybrid.cpp -o cartoon_hybrid.exe

## Ejecución

### Generar una imagen de prueba

./generate_test_image.exe output.bmp

### Ejecutar la versión secuencial

./cartoon_sequential.exe input.bmp output.bmp 3 9 60

Argumentos:

- `input.bmp`: imagen de entrada
- `output.bmp`: imagen generada
- `filterSize`: 3 o 5
- `levels`: cantidad de niveles de posterización (ejemplo: 3 o 9)
- `threshold`: umbral para la detección de bordes

### Ejecutar MPI (ejemplo)

C:\msys64\ucrt64\bin\mpiexec.exe -n 4 ./cartoon_mpi.exe input.bmp output.bmp 3 9 60

> En Windows puede ser necesario usar la ruta completa del ejecutable de `mpiexec.exe` si el launcher no está en el PATH.

## Descripción del algoritmo

1. Se carga la imagen BMP en memoria como una matriz de pixeles RGB.
2. Se aplica un filtro de suavizado para reducir ruido.
3. Se detectan bordes usando un operador tipo Sobel sobre la imagen suavizada.
4. Se realiza posterización por canal en R, G y B.
5. Se combinan bordes y colores para obtener la apariencia cartoon.

## Observación

Este proyecto está pensado para benchmarking y comparación de rendimiento entre diferentes configuraciones paralelas, siguiendo la consigna del trabajo práctico.
