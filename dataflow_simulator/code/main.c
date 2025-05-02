#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "common.h"
#include "string.h"
#include "top.h"
#include <fftw3.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

float2 make_float2(float x, float y) {
	float2 result;
	result.x = x;
	result.y = y;
	return result;
}
int2 make_int2(int x,int y)
{
	int2 int2_data;

	int2_data.x=x;
	int2_data.y=y;

	return(int2_data);

}
PRECISION2 complex_mult_CPU(const PRECISION2 z1, const PRECISION2 z2)
{
	return MAKE_PRECISION2(z1.x * z2.x - z1.y * z2.y, z1.y * z2.x + z1.x * z2.y);
}
void gridding_CPU(PRECISION2 *grid, const PRECISION2 *kernel, const int2 *supports,
                  const PRECISION3 *vis_uvw, const PRECISION2 *vis, const int num_vis, const int oversampling,
                  const int grid_size, const double uv_scale, const double w_scale)
{
    // const unsigned int vis_index = blockIdx.x * blockDim.x + threadIdx.x;
	//double w_scale_bis = 0.0;
    // if(vis_index >= num_vis)
    // 	return;
    for (unsigned int vis_index = 0; vis_index < num_vis; vis_index++)
    {

        // Represents index of w-projection kernel in supports array
        const int plane_index = (int) ROUND(SQRT(ABS(vis_uvw[vis_index].z * w_scale)));

        // Scale visibility uvw into grid coordinate space
        const PRECISION2 grid_coord = MAKE_PRECISION2(
                vis_uvw[vis_index].x * uv_scale,
                vis_uvw[vis_index].y * uv_scale
        );
        const int half_grid_size = grid_size / 2;
        const int half_support = supports[plane_index].x;

        PRECISION conjugate = (vis_uvw[vis_index].z < 0.0) ? -1.0 : 1.0;

        const PRECISION2 snapped_grid_coord = MAKE_PRECISION2(
                ROUND(grid_coord.x * oversampling) / oversampling,
                ROUND(grid_coord.y * oversampling) / oversampling
        );

        const PRECISION2 min_grid_point = MAKE_PRECISION2(
                CEIL(snapped_grid_coord.x - half_support),
                CEIL(snapped_grid_coord.y - half_support)
        );

        const PRECISION2 max_grid_point = MAKE_PRECISION2(
                FLOOR(snapped_grid_coord.x + half_support),
                FLOOR(snapped_grid_coord.y + half_support)
        );
        // PRECISION2 grid_point = MAKE_PRECISION2(0.0, 0.0);
        PRECISION2 convolved = MAKE_PRECISION2(0.0, 0.0);
        PRECISION2 kernel_sample = MAKE_PRECISION2(0.0, 0.0);
        int2 kernel_uv_index = make_int2(0, 0);

        int grid_index = 0;
        int kernel_index = 0;
        int w_kernel_offset = supports[plane_index].y;

        // printf("%lf \t %lf\n", max_grid_point.x - min_grid_point.x, max_grid_point.y - min_grid_point.y);

        for(int grid_v = min_grid_point.y; grid_v <= max_grid_point.y; ++grid_v)
        {
        	if(grid_v < -half_grid_size || grid_v >= half_grid_size){
				continue;
			}

            kernel_uv_index.y = abs((int)ROUND((grid_v - snapped_grid_coord.y) * oversampling));

            for(int grid_u = min_grid_point.x; grid_u <= max_grid_point.x; ++grid_u)
            {
            	if(grid_u < -half_grid_size || grid_u >= half_grid_size){
					continue;
				}

                kernel_uv_index.x = abs((int)ROUND((grid_u - snapped_grid_coord.x) * oversampling));

                kernel_index = w_kernel_offset + kernel_uv_index.y * (half_support + 1)
                                                 * oversampling + kernel_uv_index.x;
                kernel_sample = MAKE_PRECISION2(kernel[kernel_index].x, kernel[kernel_index].y  * conjugate);

                grid_index = (grid_v + half_grid_size) * grid_size + (grid_u + half_grid_size);

                convolved = complex_mult_CPU(vis[vis_index], kernel_sample);

                grid[grid_index].x += convolved.x;
                grid[grid_index].y += convolved.y;
                // atomicAdd(&(grid[grid_index].x), convolved.x);
                // atomicAdd(&(grid[grid_index].y), convolved.y);
            }
        }
    }

    printf("FINISHED GRIDDING\n");
}


void CUFFT_EXECUTE_FORWARD_C2C_actor(int GRID_SIZE, PRECISION2 *uv_grid_in, PRECISION2 *uv_grid_out)
{
	printf("UPDATE >>> Performing FFT...\n\n");

	//memcpy(uv_grid_out, uv_grid_in, sizeof(PRECISION2) * GRID_SIZE * GRID_SIZE);
	//return;

#if SINGLE_PRECISION

	// Need to link with -lfftw3f instead of or in addition to -lfftw3
	fftwf_plan fft_plan;

	fft_plan = fftwf_plan_dft_2d(GRID_SIZE, GRID_SIZE, (float (*)[2]) uv_grid_in, (float (*)[2]) uv_grid_out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftwf_execute(fft_plan);
	fftwf_destroy_plan(fft_plan);

#else

	fftw_plan fft_plan;

	fft_plan = fftw_plan_dft_2d(GRID_SIZE, GRID_SIZE, (double (*)[2]) uv_grid_in, (double (*)[2]) uv_grid_out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(fft_plan);
	fftw_destroy_plan(fft_plan);
#endif

	//memcpy(uv_grid_out, uv_grid_in, sizeof(PRECISION2) * GRID_SIZE * GRID_SIZE);

	//MD5_Update(sizeof(PRECISION2) * GRID_SIZE*GRID_SIZE, uv_grid_out);

	// Normalisation après la FFT
	for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
		uv_grid_out[i].x /= GRID_SIZE;
		uv_grid_out[i].y /= GRID_SIZE;
	}


}
void fft_shift_complex_to_real_actor(int GRID_SIZE, PRECISION2 *uv_grid, Config *config, PRECISION *dirty_image) {
	int grid_square = GRID_SIZE * GRID_SIZE;
	int row_index,col_index;

	printf("UPDATE >>> Shifting grid data for FFT...\n\n");
	// Perform 2D FFT shift back
	for (row_index = 0; row_index < GRID_SIZE; row_index++)
	{
		for (col_index = 0; col_index < GRID_SIZE; col_index++)
		{
			int a = 1 - 2 * ((row_index + col_index) & 1);
			dirty_image[row_index * GRID_SIZE + col_index] = uv_grid[row_index * GRID_SIZE + col_index].x * a;
		}
	}

}
void fft_shift_real_to_complex_actor(int GRID_SIZE, PRECISION *image, Config *config, PRECISION2 *fourier) {
	int grid_square = GRID_SIZE * GRID_SIZE;
	int row_index,col_index;

	printf("UPDATE >>> Shifting grid data for FFT...\n\n");
	// Perform 2D FFT shift back
	for (row_index = 0; row_index < GRID_SIZE; row_index++)
	{
		for (col_index = 0; col_index < GRID_SIZE; col_index++)
		{
			int a = 1 - 2 * ((row_index + col_index) & 1);
			fourier[row_index * GRID_SIZE + col_index].x = image[row_index * GRID_SIZE + col_index] * a;
			fourier[row_index * GRID_SIZE + col_index].y = 0;

		}
	}


	//MD5_Update(sizeof(PRECISION2) * grid_square, fourier);
}
void fft_shift_complex_to_complex_actor(int GRID_SIZE, PRECISION2 *uv_grid_in, Config *config, PRECISION2 *uv_grid_out) {
	int row_index,col_index;

	printf("UPDATE >>> Shifting grid data for FFT...\n\n");
	// Perform 2D FFT shift
	for (row_index = 0;row_index < GRID_SIZE; row_index++)
	{
		for (col_index = 0; col_index < GRID_SIZE; col_index++)
		{
			int a = 1 - 2 * ((row_index + col_index) & 1);
			uv_grid_out[row_index * GRID_SIZE + col_index].x = uv_grid_in[row_index * GRID_SIZE + col_index].x * a;
			uv_grid_out[row_index * GRID_SIZE + col_index].y = uv_grid_in[row_index * GRID_SIZE + col_index].y * a;
		}
	}
	//MD5_Update(sizeof(PRECISION2) * GRID_SIZE * GRID_SIZE, uv_grid_out);
}
void CUFFT_EXECUTE_INVERSE_C2C_actor(int GRID_SIZE, PRECISION2 *uv_grid_in, PRECISION2 *uv_grid_out)
{
	printf("UPDATE >>> Performing iFFT...\n\n");

	//memcpy(uv_grid_out, uv_grid_in, sizeof(PRECISION2) * GRID_SIZE * GRID_SIZE);
	//return;

	//memset(uv_grid_out, 0, sizeof(PRECISION2) * GRID_SIZE * GRID_SIZE);

#if SINGLE_PRECISION

	// Need to link with -lfftw3f instead of or in addition to -lfftw3
	fftwf_plan fft_plan;

	fft_plan = fftwf_plan_dft_2d(GRID_SIZE, GRID_SIZE, (float (*)[2]) uv_grid_in, (float (*)[2]) uv_grid_out, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftwf_execute(fft_plan);
	fftwf_destroy_plan(fft_plan);

#else

	fftw_plan fft_plan;

	fft_plan = fftw_plan_dft_2d(GRID_SIZE, GRID_SIZE, (double (*)[2]) uv_grid_in, (double (*)[2]) uv_grid_out, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(fft_plan);
	fftw_destroy_plan(fft_plan);
#endif

	//memcpy(uv_grid_out, uv_grid_in, sizeof(PRECISION2) * GRID_SIZE * GRID_SIZE);

	// Normalisation après la iFFT
	for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
		uv_grid_out[i].x /= GRID_SIZE;
		uv_grid_out[i].y /= GRID_SIZE;
	}

}

PRECISION2 complex_mult(const PRECISION2 z1, const PRECISION2 z2)
{
	return make_float2(z1.x * z2.x - z1.y * z2.y, z1.y * z2.x + z1.x * z2.y);
}

float randn(double mean, double stddev) {
	double u1 = ((double)rand() / RAND_MAX);
	double u2 = ((double)rand() / RAND_MAX);
	double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
	return (float)(mean + z0 * stddev);
}
double rayleigh(double sigma) {
	double u = (double) rand() / RAND_MAX;
	return sigma * sqrt(-2.0 * log(1.0 - u));
}
// Fonction de fenêtre : approximation d'une spheroidal prolate
float pswf(float x) {
	if (fabs(x) > 1.0) return 0.0;
	return powf(cosf(PI * x / 2.0f), 4); // simple approximation
}
// Fonction de Bessel modifiée d'ordre 0 (I0)
float I0(float x) {
	float sum = 1.0f;
	float term = 1.0f;
	for (int n = 1; n < 20; ++n) {
		term *= (x / (2.0f * n));
		sum += term * term;
	}
	return sum;
}

// Fonction de fenêtre : Kaiser-Bessel
float kaiser_bessel(float x, float beta) {
	if (fabs(x) > 1.0f) return 0.0f;
	float t = sqrtf(1.0f - x * x);
	return powf(I0(beta * t) / I0(beta), 2.0f); // carré pour lisser davantage
}
void generate_kernel(int NUM_KERNELS, int OVERSAMPLING_FACTOR, int KERNEL_SUPPORT,
	PRECISION2* kernels, int2* kernel_supports) {
	bool bessel_boolean = false;
	int kernel_half = KERNEL_SUPPORT/2;
	int kernel_size = 2 * kernel_half;

	int kernel_1d_len = (kernel_half + 1) * OVERSAMPLING_FACTOR;
	int kernel_2d_len = kernel_1d_len * kernel_1d_len;

	int kernel_offset = 0;

	for (int w = 0; w < NUM_KERNELS; ++w)
	{
		// Stocke le support et le décalage
		kernel_supports[w].x = kernel_half;
		kernel_supports[w].y = kernel_offset;

		// Génère le noyau 1D
		float* kernel_1d = malloc(sizeof(float) * kernel_1d_len);
		for (int i = 0; i < kernel_1d_len; ++i) {
			float x = (float)i / (float)OVERSAMPLING_FACTOR;
			float norm_x = x / (float)(kernel_half + 1);
			if (bessel_boolean) {
				float beta  =3.0f; // bessel parameter to control lobe shape: large β -> thinner lobe (better concentration but increase secondary lobes), tradeoff resolution and noise
				kernel_1d[i] = kaiser_bessel(norm_x, beta);
			}else {
				kernel_1d[i] = pswf(norm_x);
			}


		}

		// Normalise le noyau 1D
		float sum = 0.f;
		for (int i = 0; i < kernel_1d_len; ++i) sum += kernel_1d[i];
		for (int i = 0; i < kernel_1d_len; ++i) kernel_1d[i] /= sum;

		// Produit tensoriel pour obtenir le noyau 2D
		for (int y = 0; y < kernel_1d_len; ++y) {
			for (int x = 0; x < kernel_1d_len; ++x) {
				int idx = kernel_offset + y * kernel_1d_len + x;
				float val = kernel_1d[x] * kernel_1d[y];
				kernels[idx].x = val;
				kernels[idx].y = 0.0f;
			}
		}

		kernel_offset += kernel_2d_len;
		free(kernel_1d);
	}
	printf("kernel_offset %d\n",kernel_offset);

	printf("UPDATE >>> Image degridded successfully\n");
}

void std_degridding(int GRID_SIZE, int NUM_VISIBILITIES, int NUM_KERNELS, int TOTAL_KERNEL_SAMPLES, int OVERSAMPLING_FACTOR, PRECISION2* kernels,
		int2* kernel_supports, PRECISION2* input_grid, PRECISION3* vis_uvw_coords, int* num_corrected_visibilities, Config* config,
		PRECISION2* output_visibilities){
	printf("Degridding visibilities using FFT degridder\n");

	memset(output_visibilities, 0, sizeof(PRECISION2) * NUM_VISIBILITIES);

	int grid_center = GRID_SIZE / 2;

	for(int i = 0; i < *num_corrected_visibilities; ++i){
		// Calcul de l'indice w basé sur la coordonnée z des visibilités corrigées
		int w_idx = (int)(SQRT(ABS(vis_uvw_coords[i].z * config->w_scale)) + 0.5);
		// we don't care first on w
		//w_idx = 0;

		// Récupère l'épaisseur du noyau et le décalage pour le plan w
		int half_support = kernel_supports[w_idx].x;
		int w_offset = kernel_supports[w_idx].y;

		// Calcul de la position sur la grille à partir des coordonnées UV
		PRECISION2 grid_pos = {.x = vis_uvw_coords[i].x * config->uv_scale, .y = vis_uvw_coords[i].y * config->uv_scale};
		//printf("i: %d, grid_pos: (%f, %f)\n", i, grid_pos.x, grid_pos.y);

		// Détermine si la visibilité doit être conjuguée selon la coordonnée z
		PRECISION conjugate = (vis_uvw_coords[i].z < 0.0) ? -1.0 : 1.0;

		// Initialisation de la norme des coefficients du noyau pour la normalisation
		float comm_norm = 0.f;

        // Parcours des positions sur l'axe y dans la fenêtre du noyau
        for(int v = CEIL(grid_pos.y - half_support); v < CEIL(grid_pos.y + half_support); ++v)
		{
        	// Correction de l'indice v si nécessaire pour respecter les bornes de la grille
			int corrected_v = v;
			if(v < -grid_center){
				corrected_v = (-grid_center - v) - grid_center;
			}
			else if(v >= grid_center){
				corrected_v = (grid_center - v) + grid_center;
			}
        	// Clamp final pour éviter tout débordement
        	corrected_v = MAX(-grid_center, MIN(grid_center - 1, corrected_v));

        	// Détermination de l'indice du noyau sur l'axe y
			int2 kernel_idx;
			kernel_idx.y = abs((int)ROUND((corrected_v - grid_pos.y) * OVERSAMPLING_FACTOR));

        	// Parcours des positions sur l'axe x dans la fenêtre du noyau
			for(int u = CEIL(grid_pos.x - half_support); u < CEIL(grid_pos.x + half_support); ++u)
			{
				int corrected_u = u;

				if(u < -grid_center){
					corrected_u = (-grid_center - u) - grid_center;
				}
				else if(u >= grid_center){
					corrected_u = (grid_center - u) + grid_center;
				}

				// Clamp final pour éviter tout débordement
				corrected_u = MAX(-grid_center, MIN(grid_center - 1, corrected_u));

				// Détermination de l'indice du noyau sur l'axe x
				kernel_idx.x = abs((int)ROUND((corrected_u - grid_pos.x) * OVERSAMPLING_FACTOR));

				// Calcul de l'indice du noyau dans le tableau des noyaux
				int k_idx = w_offset + kernel_idx.y * (half_support + 1) * OVERSAMPLING_FACTOR + kernel_idx.x;

				// Récupère l'échantillon du noyau et applique la conjugaison si nécessaire
				PRECISION2 kernel_sample = kernels[k_idx];

				kernel_sample.y *= conjugate;

				// Calcul de l'indice de la grille d'entrée pour récupérer la valeur
				int grid_idx = (corrected_v + grid_center) * GRID_SIZE + (corrected_u + grid_center);
				//printf("grid_idx: %d, input_grid[grid_idx]: (%f, %f)\n", grid_idx, input_grid[grid_idx].x, input_grid[grid_idx].y);

				// Effectue le produit complexe entre la valeur de la grille et l'échantillon du noyau
				PRECISION2 prod = complex_mult(input_grid[grid_idx], kernel_sample);

				// Accumule la norme des échantillons du noyau pour la normalisation
				comm_norm += kernel_sample.x * kernel_sample.x + kernel_sample.y * kernel_sample.y;

				// Ajoute le produit au résultat final de la visibilité
				output_visibilities[i].x += prod.x;
				output_visibilities[i].y += prod.y;
			}
		}

		// Calcul de la norme des noyaux pour la normalisation
		comm_norm = sqrt(comm_norm);

		// Normalisation des visibilités pour éviter les valeurs trop petites
		output_visibilities[i].x = comm_norm < 1e-5f ? output_visibilities[i].x / 1e-5f : output_visibilities[i].x / comm_norm;
		output_visibilities[i].y = comm_norm < 1e-5f ? output_visibilities[i].y / 1e-5f : output_visibilities[i].y / comm_norm;

	}
	printf("UPDATE >>> Image degridded successfully\n");
}


void load_image_from_file(PRECISION *input_grid, Config* config) {
	const char *file_name = config->final_image_output;
	FILE *f = fopen(file_name, "r");
	if (f == NULL) {
		perror(">>> ERROR");
		printf(">>> ERROR: Unable to open file %s...\n\n", file_name);
		return;
	}

	int i = 0;
	 while (fscanf(f, "%f%*[, \n]", &input_grid[i]) == 1) {
		i++;
		if (i >= 512*512) {
			printf(">>> WARNING: Too many elements in file, truncating\n");
			break;
		}
	}
	fclose(f);
	printf("UPDATE >>> Image loaded from %s, %d elements \n\n", file_name, i);
}




void handle_file_error(FILE *file, const char *message) {
	if (file) {
		fclose(file);  // Ferme le fichier si ouvert
	}
	perror(message);
	exit(EXIT_FAILURE);
}

void convert_vis_to_csv(int NUM_VISIBILITIES, PRECISION2* output_visibilities, PRECISION3* corrected_vis_uvw_coords, Config *config) {
	FILE* file = fopen(config->visibility_source_file, "w");
	if (file == NULL) {
		handle_file_error(file, "Erreur lors de l'ouverture du fichier");
	}

	// Écrire le nombre de visibilités dans la première ligne
	if (fprintf(file, "%d\n", NUM_VISIBILITIES) < 0) {
		handle_file_error(file, "Erreur lors de l'écriture du nombre de visibilités");
	}

	for (int i = 0; i < NUM_VISIBILITIES; i++) {
		// Vérification des valeurs NaN
		if (isnan(corrected_vis_uvw_coords[i].x) || isnan(corrected_vis_uvw_coords[i].y) || isnan(corrected_vis_uvw_coords[i].z) ||
			isnan(output_visibilities[i].x) || isnan(output_visibilities[i].y)) {
			fprintf(stderr, "Erreur : des données invalides (NaN) rencontrées à l'index %d.\n", i);
			handle_file_error(file, "Erreur lors de la rencontre de données invalides");
			}

		// Écriture des données dans le fichier CSV
		if (fprintf(file, "%.6f %.6f %.6f %.6f %.6f 1\n",
					corrected_vis_uvw_coords[i].x,
					corrected_vis_uvw_coords[i].y,
					corrected_vis_uvw_coords[i].z,
					output_visibilities[i].x,
					output_visibilities[i].y) < 0) {
			handle_file_error(file, "Erreur lors de l'écriture dans le fichier");
					}
	}

	// Fermer le fichier
	if (fclose(file) != 0) {
		handle_file_error(file, "Erreur lors de la fermeture du fichier");
	}

	printf("UPDATE >>> write csv to %s\n\n", config->visibility_source_file);
}



void vis_coord_set_up(int NUM_VISIBILITIES, int GRID_SIZE, int TIMING_SAMPLE, PRECISION3* vis_uvw_coords, Config *config) {
	const float MAX_UVW = config->max_w;
	double meters_to_wavelengths = SPEED_OF_LIGHT / config->frequency_hz;  // Conversion de mètre en longueurs d'onde

	const double sigma = MAX_UVW / 30.0;  // Dispersion pour l’axe Z
	int points_per_circle = 10;  // Nombre initial de points par cercle

	int current_num_points = 0;
	int r = 1;

	while (current_num_points < NUM_VISIBILITIES) {
		double radius = r * MAX_UVW / 10.0;  // Rayon du cercle

		for (int p = 0; p < points_per_circle; p++) {
			if (current_num_points >= NUM_VISIBILITIES) break;  // Vérification à chaque itération

			double angle = (2.0 * M_PI / points_per_circle) * p;
			float x = (float)(radius * cos(angle));
			float y = (float)(radius * sin(angle));
			float z = (float)randn(0.0, sigma);  // Bruit Gaussien sur Z

			vis_uvw_coords[current_num_points].x = x * meters_to_wavelengths;
			vis_uvw_coords[current_num_points].y = y * meters_to_wavelengths;
			vis_uvw_coords[current_num_points].z = z * meters_to_wavelengths;

			current_num_points++;
		}

		r++;  // Augmente le rayon
		points_per_circle += 5;  // Augmente la densité des points
	}

	printf("Points de coordonnées de visibilités générés : %d (attendus : %d)\n\n", current_num_points, NUM_VISIBILITIES);
}

void export_image_to_csv(PRECISION2 *image_grid, int grid_size, const char *output_csv_file) {
	FILE *f = fopen(output_csv_file, "w");
	if (f == NULL) {
		perror(">>> ERROR");
		printf(">>> ERROR: Unable to open file %s...\n\n", output_csv_file);
		return;
	}

	// Écrire les données complexes dans le fichier CSV
	for (int i = 0; i < grid_size*grid_size; i++) {
		fprintf(f, "%.6f, %.6f\n", image_grid[i].x, image_grid[i].y); // Partie réelle, partie imaginaire
	}

	printf("Image saved to CSV: %s\n", output_csv_file);

	// Fermer le fichier
	fclose(f);
}
void export_real_to_csv(PRECISION *image_grid, int grid_size, const char *output_csv_file) {
	FILE *f = fopen(output_csv_file, "w");
	if (f == NULL) {
		perror(">>> ERROR");
		printf(">>> ERROR: Unable to open file %s...\n\n", output_csv_file);
		return;
	}

	// Écrire les données complexes dans le fichier CSV
	for (int i = 0; i < grid_size*grid_size; i++) {
		fprintf(f, "%.6f, %.6f\n", image_grid[i], 0.0); // Partie réelle, partie imaginaire
	}

	printf("Image saved to CSV: %s\n", output_csv_file);

	// Fermer le fichier
	fclose(f);
}

void save_heatmap_png(const float* image, int size, const char* filename) {
	unsigned char* pixels = malloc(size * size * sizeof(unsigned char));
	if (!pixels) {
		fprintf(stderr, "Erreur allocation mémoire heatmap\n");
		return;
	}

	// Trouver min/max pour normaliser
	float min_val = FLT_MAX, max_val = -FLT_MAX;
	for (int i = 0; i < size * size; ++i) {
		if (image[i] < min_val) min_val = image[i];
		if (image[i] > max_val) max_val = image[i];
	}

	float range = max_val - min_val;
	if (range == 0.0f) range = 1.0f; // éviter division par zéro

	// Normalisation + conversion en niveaux de gris (0-255)
	for (int i = 0; i < size * size; ++i) {
		float norm = (image[i] - min_val) / range;
		pixels[i] = (unsigned char)(norm * 255.0f);
	}

	// Sauvegarde PNG
	int success = stbi_write_png(filename, size, size, 1, pixels, size);
	if (!success) {
		fprintf(stderr, "Erreur d'écriture PNG %s\n", filename);
	} else {
		printf("✅ PNG sauvegardé : %s\n", filename);
	}

	free(pixels);
}

int main(void) {

	int FOV_DEGREES = 1; //champs de vue
	int NUM_CHANNEL = 1;//nombre de canaux de frequence
	int NUM_POL = 1;// nombre de polarisation
	int TIMING_SAMPLE = 500;

	int NUM_RECEIVERS = 5;//nombre d'antennes
	int NUM_BASELINE = NUM_RECEIVERS*(NUM_RECEIVERS-1)/2; // nombre de paire d'antennes
	int OVERSAMPLING_FACTOR = 32; // nombre de fois que les données sont surechantillonée
	int GRID_SIZE = 64; // taille de l'image fits "true sky"
	int NUM_VISIBILITIES = NUM_BASELINE*TIMING_SAMPLE*NUM_CHANNEL*NUM_POL; // greater than (GRID_SIZE/cell_size)²
	int NUM_KERNEL = OVERSAMPLING_FACTOR-1;//nombre de noyaux de convolution
	int KERNEL_SUPPORT = 2;
	int k = 2;//GRID_SIZE doit etr au moins 2 fois superieur à la taille du noyau kernel_size = 2*half+1
	int half_support =KERNEL_SUPPORT/2;//(int)( (GRID_SIZE/2 -1)/k);
	int TOTAL_KERNEL_SAMPLES = NUM_KERNEL * pow((half_support + 1) * OVERSAMPLING_FACTOR,2);//(int)(pow(2*half_support*OVERSAMPLING_FACTOR,2)*NUM_KERNEL); //c'est recalculé dans le processus mais faut mettre une taille suffisante



	PRECISION2* kernels = (PRECISION2*)malloc(sizeof(PRECISION2) * TOTAL_KERNEL_SAMPLES);
	int2* kernel_supports = (int2*)malloc(sizeof(int2) * NUM_KERNEL);
	PRECISION* input_grid = (PRECISION*)malloc(sizeof(PRECISION) * GRID_SIZE * GRID_SIZE);
	PRECISION2* input_grid_shift = (PRECISION2*)malloc(sizeof(PRECISION2) * GRID_SIZE * GRID_SIZE);

	PRECISION3* vis_uvw_coords = (PRECISION3*)malloc(sizeof(PRECISION3) * NUM_VISIBILITIES);
	int* num_corrected_visibilities = (int*)malloc(sizeof(int) * 1);
	PRECISION2* output_visibilities = (PRECISION2*)malloc(sizeof(PRECISION2) * NUM_VISIBILITIES);
	Config config;
	PRECISION2* uv_grid = (PRECISION2*)malloc(sizeof(PRECISION2) * GRID_SIZE * GRID_SIZE);
	PRECISION2* uv_grid_shift = (PRECISION2*)malloc(sizeof(PRECISION2) * GRID_SIZE * GRID_SIZE);
	PRECISION2* output_grid = (PRECISION2*)malloc(sizeof(PRECISION2) * GRID_SIZE * GRID_SIZE);
	PRECISION* output_grid_shift = (PRECISION*)malloc(sizeof(PRECISION) * GRID_SIZE * GRID_SIZE);



	//initialize values (pas necessaire mais au cas ou)

	memset(kernels, 0, sizeof(PRECISION2)*TOTAL_KERNEL_SAMPLES);
	memset(kernel_supports, 0, sizeof(int2) * NUM_KERNEL);
	memset(input_grid, 0, sizeof(PRECISION) * GRID_SIZE * GRID_SIZE);
	memset(vis_uvw_coords, 0, sizeof(PRECISION3) * NUM_VISIBILITIES);
	memset(output_visibilities, 0, sizeof(PRECISION2) * NUM_VISIBILITIES);



	for(int i = 0; i < 1; ++i){
		num_corrected_visibilities[i] = NUM_VISIBILITIES;
	}

	if (kernels == NULL || kernel_supports == NULL || input_grid == NULL ||
	vis_uvw_coords == NULL  ||
	num_corrected_visibilities == NULL || output_visibilities == NULL) {
		fprintf(stderr, "Erreur d'allocation mémoire\n");
		exit(EXIT_FAILURE);
	}

	config.frequency_hz = SPEED_OF_LIGHT/0.21; // observed frequency --> osef
	config.baseline_max = 800;
	config.max_w = config.baseline_max*config.frequency_hz/SPEED_OF_LIGHT;// baseline_max * freq obs --> define the frequency boundaries
	printf("max_w: %.6f\n", config.max_w);
	config.w_scale = pow(NUM_KERNEL - 1, 2.0) / config.max_w;
	printf("w_scale: %.6f\n", config.w_scale);
	config.cell_size = (FOV_DEGREES * M_PI) / (180.0 * GRID_SIZE);// lower than 1/2.f_max
	config.uv_scale =  1*config.cell_size*GRID_SIZE;

	printf("uv_scale: %.6f\n", config.uv_scale);

	config.num_baselines = NUM_BASELINE;
	config.num_kernels = NUM_KERNEL;
	config.total_kernel_samples = TOTAL_KERNEL_SAMPLES;

	config.visibility_source_file = "vis.csv";
	config.output_path = "";
	config.final_image_output = "image.csv";
	config.degridding_kernel_support_file = "config/kernel_support.csv";
	config.degridding_kernel_imag_file = "config/kernel_imag.csv";
	config.degridding_kernel_real_file= "config/kernel_real.csv";
	config.degridding_kernel_support_file = "config/w-proj_supports_x16_2458_image.csv";
	config.degridding_kernel_imag_file = "config/w-proj_kernels_imag_x16_2458_image.csv";
	config.degridding_kernel_real_file= "config/w-proj_kernels_real_x16_2458_image.csv";

	config.oversampling = OVERSAMPLING_FACTOR;

	// Lecture d'une image de référence (ciel réel ou simulé) au format CSV.
	// Cette image représente l'intensité du ciel dans le domaine image (l,m).
	load_image_from_file( input_grid, &config);


	// Simulation d'un jeu de visibilités en générant leurs positions dans le plan UVW,
	// basé sur des paramètres comme le nombre de visibilités et la taille de grille.
	vis_coord_set_up(NUM_VISIBILITIES, GRID_SIZE, TIMING_SAMPLE,vis_uvw_coords, &config);

	// Calcul des facteurs d'échelle et préparation des noyaux utilisés pour
	// projeter/interpoler les visibilités sur la grille UV.
	//degridding_kernel_host_set_up( NUM_KERNEL, TOTAL_KERNEL_SAMPLES, &config, kernel_supports,  kernels);
	generate_kernel(NUM_KERNEL,OVERSAMPLING_FACTOR,KERNEL_SUPPORT,kernels,kernel_supports);

	//option: genere png de l'image d'entrée
	save_heatmap_png(input_grid, GRID_SIZE, "input_heatmap.png");

	// Décalage du centre de l'image (FFT shift) pour que l'origine (0,0) soit au centre.
	// Utile avant d'appliquer la transformée de Fourier.
	fft_shift_real_to_complex_actor(GRID_SIZE,input_grid,&config,input_grid_shift);
	export_image_to_csv(input_grid_shift, GRID_SIZE, "input_grid.csv");


	// Passage en Fourier (I(l,m)->V(u,v))
	CUFFT_EXECUTE_FORWARD_C2C_actor(GRID_SIZE, input_grid_shift, uv_grid);
	export_image_to_csv(uv_grid, GRID_SIZE, "uv_grid.csv");

	// Nouvelle opération de shift pour centrer les visibilités dans le plan UV.
	fft_shift_complex_to_complex_actor(GRID_SIZE,uv_grid,&config,uv_grid_shift);
	export_image_to_csv(uv_grid_shift, GRID_SIZE, "uv_grid_shift.csv");

	// Extraction des valeurs de visibilités à des positions précises (u,v,w)
	// à partir de la grille UV via interpolation (dégridding).
	std_degridding(GRID_SIZE, NUM_VISIBILITIES, NUM_KERNEL, TOTAL_KERNEL_SAMPLES,OVERSAMPLING_FACTOR, kernels, kernel_supports, uv_grid_shift,vis_uvw_coords, num_corrected_visibilities, &config, output_visibilities);

	// Enregistrement des visibilités simulées dans un fichier CSV.
	convert_vis_to_csv(NUM_VISIBILITIES,output_visibilities, vis_uvw_coords, &config);

	// Processus inverse du dégridding : on projette les visibilités reconstruites sur une grille UV.
	gridding_CPU(uv_grid, kernels, kernel_supports,
	vis_uvw_coords, output_visibilities, NUM_VISIBILITIES, OVERSAMPLING_FACTOR,
	GRID_SIZE, config.uv_scale, config.w_scale);


	// Reconstruction avec iFFT
	CUFFT_EXECUTE_INVERSE_C2C_actor(GRID_SIZE, uv_grid, output_grid);
	export_image_to_csv(output_grid, GRID_SIZE, "reconstructed.csv");

	// Dernier shift pour recentrer l'image en domaine image + export en CSV et image PNG.
	fft_shift_complex_to_real_actor(GRID_SIZE,output_grid,&config,output_grid_shift);
	export_real_to_csv(output_grid_shift, GRID_SIZE, "reconstructed_shift.csv");
	save_heatmap_png(output_grid_shift, GRID_SIZE, "reconstructed_heatmap.png");

	free(kernel_supports);
	kernel_supports = NULL;
	free(kernels);
	kernels = NULL;
	free(input_grid);
	input_grid = NULL;
	free(output_visibilities);
	output_visibilities = NULL;
	free(num_corrected_visibilities);
	num_corrected_visibilities = NULL;
	free(vis_uvw_coords);
	vis_uvw_coords = NULL;
	free(uv_grid);
	uv_grid = NULL;
	free(uv_grid_shift);
	uv_grid_shift = NULL;
	free(output_grid);
	output_grid = NULL;
	free(input_grid_shift);
	input_grid_shift = NULL;
	free(output_grid_shift);
	output_grid_shift = NULL;

	return 0;
}

/*// Affichage des résultats
for (int i = 0; i < 5; i++) {
	printf("kernel_supports %d: %d + %di\n", i, kernel_supports[i].x, kernel_supports[i].y);
}
for (int i = 0; i < 5; i++) {
	printf("kernels %d: %.6f + %.6fi\n", i, kernels[i].x, kernels[i].y);
}*/
