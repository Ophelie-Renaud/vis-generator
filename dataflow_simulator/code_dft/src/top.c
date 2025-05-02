#include "top.h"
#include <math.h>

void end_sink(__attribute__((unused)) int NUM_RECEIVERS, __attribute__((unused)) PRECISION2 *gains)
{
	return;
}

void config_struct_set_up(int GRID_SIZE, int NUM_KERNELS, OUT Config *config_struct)
{
	config_struct->gpu_max_threads_per_block           = 1024;
	config_struct->gpu_max_threads_per_block_dimension = 32;
	//radians per cell, corresponds with 1 degree fov for an image of 2048x2048, not sure why actual image is set to 2458, investigate
	config_struct->cell_size                           = 8.52211548825356E-06;
	config_struct->frequency_hz                        = SPEED_OF_LIGHT;
	config_struct->dirty_image_output                   = "gleam_small_dirty_image.csv";
	config_struct->right_ascension						= true;
	config_struct->visibility_source_file				= "data/input/GLEAM_small_visibilities_corrupted.csv";
	//    config_struct->visibility_source_file				= "../data/input/gleam_small_visibilities_corrupted_0p5_std.csv";

	config_struct->output_path							= "data/output/small/";

	// Testing
	config_struct->perform_system_test                 = false;
	config_struct->system_test_image                   = "";
	config_struct->system_test_sources                 = "";
	config_struct->system_test_visibilities            = "";

	// Gains - NOTE: initial gains from file logic not implemented yet
	config_struct->default_gains_file = "data/input/TrueGainsNotRotated.csv";
	config_struct->output_gains_file = "estimated_gains.csv";
	config_struct->use_default_gains	= true;

	// Gridding
	config_struct->max_w               = 1895.410847844;
	config_struct->w_scale             = pow(NUM_KERNELS - 1, 2.0) / config_struct->max_w;
	config_struct->oversampling        = 16;
	config_struct->uv_scale            = config_struct->cell_size * GRID_SIZE;
	config_struct->kernel_real_file    = "data/input/kernels/new/wproj_manualconj_degridding_kernels_real_x16.csv";
	config_struct->kernel_imag_file    = "data/input/kernels/new/wproj_manualconj_degridding_kernels_imag_x16.csv";
	config_struct->kernel_support_file = "data/input/kernels/new/wproj_manualconj_degridding_kernel_supports_x16.csv";
	config_struct->degridding_kernel_real_file    = "data/input/kernels/new/wproj_manualconj_degridding_kernels_real_x16.csv";
	config_struct->degridding_kernel_imag_file    = "data/input/kernels/new/wproj_manualconj_degridding_kernels_imag_x16.csv";
	config_struct->degridding_kernel_support_file = "data/input/kernels/new/wproj_manualconj_degridding_kernel_supports_x16.csv";
	config_struct->force_weight_to_one	= true;

	// Deconvolution
	config_struct->loop_gain           = 0.1;  // 0.1 is typical
	config_struct->weak_source_percent_gc = 0.005;//0.00005; // example: 0.05 = 5%
	config_struct->weak_source_percent_img = 0.0002;//0.00005; // example: 0.05 = 5%
	config_struct->psf_max_value       = 0.0;  // customize as needed, or allow override by reading psf.
	/*
		Used to determine if we are extracting noise, based on the assumption
		that located source < noise_detection_factor * running_average
	*/
	config_struct->noise_factor          = 1.5;
	config_struct->model_sources_output  = "sample_model_sources.csv";
	config_struct->residual_image_output = "residual_image.csv";
	config_struct->psf_input_file        = "data/input/gleam_small_psf.csv";

	config_struct->psf_image_output = "dirty_psf.csv";
	config_struct->clean_psf_image_output = "clean_psf.csv";
	config_struct->model_image_output = "model.csv";
	config_struct->final_image_output = "deconvolved.csv";

	// Direct Fourier Transform
	config_struct->predicted_vis_output = "predicted_visibilities.csv";
}

void config_struct_set_up_v2(int GRID_SIZE, int NUM_KERNELS, int NUM_BASELINES, int TOTAL_KERNEL_SAMPLES, int OVERSAMPLING_FACTOR,OUT Config *config_struct) {
	// load parameter
	config_struct->num_baselines = NUM_BASELINES;
	config_struct->num_kernels = NUM_KERNELS;
	config_struct->total_kernel_samples = 0;
	config_struct->grid_size = GRID_SIZE;
	config_struct->oversampling = OVERSAMPLING_FACTOR;

	// define other constant
	float fov_degrees = 1;
	int baseline_max = 1000;

	//
	config_struct->frequency_hz = SPEED_OF_LIGHT/0.21; // observed frequency --> osef
	config_struct->max_w = baseline_max*config_struct->frequency_hz/SPEED_OF_LIGHT;// baseline_max * freq obs /celerite mais osef en fait
	config_struct->w_scale = pow(NUM_KERNELS - 1, 2.0) / config_struct->max_w;
	config_struct->cell_size = (fov_degrees * PI) / (180.0 * GRID_SIZE);// lower than 1/2.f_max
	config_struct->uv_scale =  config_struct->cell_size*GRID_SIZE;

	config_struct->loop_gain           = 0.1;  // 0.1 is typical
	config_struct->weak_source_percent_gc = 0.005;//0.00005; // example: 0.05 = 5%
	config_struct->weak_source_percent_img = 0.0002;//0.00005; // example: 0.05 = 5%
	config_struct->psf_max_value       = 0.0;  // customize as needed, or allow override by reading psf.
	config_struct->noise_factor          = 1.5;

	// Gains - NOTE: initial gains from file logic not implemented yet
	config_struct->default_gains_file = "data/input/TrueGainsNotRotated.csv";
	config_struct->output_gains_file = "estimated_gains.csv";
	config_struct->use_default_gains	= true;
	config_struct->force_weight_to_one	= true;

	// input files
	config_struct->visibility_source_file = "../code/vis.csv";
	config_struct->visibility_source_file				= "data/input/GLEAM_small_visibilities_corrupted.csv";
	config_struct->kernel_real_file    = "../code/config/wproj_manualconj_gridding_kernel_supports_x16.csv";
	config_struct->kernel_imag_file    = "../code/config/wproj_manualconj_gridding_kernels_imag_x16.csv";
	config_struct->kernel_support_file = "../code/config/wproj_manualconj_gridding_kernels_real_x16.csv";
	config_struct->degridding_kernel_support_file = "../code/config/wproj_manualconj_degridding_kernel_supports_x16.csv";
	config_struct->degridding_kernel_imag_file = "../code/config/wproj_manualconj_degridding_kernels_imag_x16.csv";
	config_struct->degridding_kernel_real_file= "../code/config/wproj_manualconj_degridding_kernels_real_x16.csv";

	config_struct->kernel_real_file =
config_struct->degridding_kernel_real_file = "data/input/kernels/new/wproj_manualconj_degridding_kernels_real_x16.csv";

	config_struct->kernel_imag_file =
	config_struct->degridding_kernel_imag_file = "data/input/kernels/new/wproj_manualconj_degridding_kernels_imag_x16.csv";

	config_struct->kernel_support_file =
	config_struct->degridding_kernel_support_file = "data/input/kernels/new/wproj_manualconj_degridding_kernel_supports_x16.csv";


	config_struct->kernel_real_file =
config_struct->degridding_kernel_real_file = "../code/config/kernel_real.csv";

	config_struct->kernel_imag_file =
	config_struct->degridding_kernel_imag_file = "../code/config/kernel_imag.csv";

	config_struct->kernel_support_file =
	config_struct->degridding_kernel_support_file = "../code/config/kernel_support.csv";


	config_struct->psf_input_file        = "data/input/psf.csv";


	// output files
	config_struct->output_path	= "data/output/tune/";
	config_struct->psf_image_output = "dirty_psf.csv";
	config_struct->clean_psf_image_output = "clean_psf.csv";
	config_struct->model_image_output = "model.csv";
	config_struct->final_image_output = "deconvolved.csv";
	config_struct->model_sources_output  = "sample_model_sources.csv";
	config_struct->residual_image_output = "residual_image.csv";

}


void gains_host_set_up(int NUM_RECEIVERS, __attribute__((unused)) int NUM_BASELINES, Config *config, PRECISION2 *gains, int2 *receiver_pairs)
{
	if(config->use_default_gains)
	{	
		for(int i = 0 ; i < NUM_RECEIVERS; ++i)
		{
			gains[i] = (PRECISION2) {.x = (PRECISION)1.0, .y = (PRECISION)0.0};
		}
	}
	else
	{
		printf(">>> UPDATE: Loading default gains from file %s ",config->default_gains_file);
		FILE *file_gains = fopen(config->default_gains_file , "r");

		if(!file_gains)
		{	
			printf(">>> ERROR: Unable to LOAD GAINS FILE grid files %s , check file structure exists...\n\n", config->default_gains_file);
			exit(EXIT_FAILURE);
			//return false;
		}
		double gainsReal = 0.0;
		double gainsImag	= 0.0;
		for(int i = 0 ; i < NUM_RECEIVERS; ++i)
		{
		// #if SINGLE_PRECISION
		// 	fscanf(file_gains, "%f %f ", &gainsReal, &gainsImag);
		// #else
			fscanf(file_gains, "%lf %lf ", &gainsReal, &gainsImag);
		//#endif

			gains[i] = (PRECISION2) {.x = (PRECISION)gainsReal, .y = (PRECISION)gainsImag};
		}
	}

	//allocate receiver pairs
	// host->receiver_pairs = (int2*) calloc(config->num_baselines, sizeof(int2));
	// if(host->receiver_pairs == NULL)
	// 	return false;

	calculate_receiver_pairs(NUM_BASELINES, NUM_RECEIVERS, receiver_pairs);

#ifdef VERBOSE_MD5
	MD5_Update(NUM_BASELINES, receiver_pairs);
#endif
	//return true;
}

void calculate_receiver_pairs(int NUM_BASELINES, int NUM_RECEIVERS, int2 *receiver_pairs)
{
	int a = 0;
	int b = 1;

	for(int i=0;i < NUM_BASELINES;++i)
	{
		//printf(">>>> CREATING RECEIVER PAIR (%d,%d) \n",a,b);
		receiver_pairs[i].x = a;
		receiver_pairs[i].y = b;

		b++;
		if(b >= NUM_RECEIVERS)
		{
			a++;
			b = a+1;
		}
	}
}


void visibility_host_set_up(int NUM_VISIBILITIES, Config *config, PRECISION3 *vis_uvw_coords, PRECISION2 *measured_vis)
{
	printf("UPDATE >>> Loading visibilities from file %s...\n\n", config->visibility_source_file);
    FILE *vis_file = fopen(config->visibility_source_file, "r");
    if(vis_file == NULL)
    {   printf("ERROR >>> Unable to open visibility file %s...\n\n", config->visibility_source_file);
		exit(EXIT_FAILURE);
        // return false; // unsuccessfully loaded data
    }
    
    // Configure number of visibilities from file
    int num_vis = 0;
    fscanf(vis_file, "%d", &num_vis);
    //config->num_visibilities = num_vis;

    // Allocate memory for incoming visibilities
    //vis_uvw_coords = (PRECISION3*) calloc(NUM_VISIBILITIES, sizeof(Visibility));
    //measured_vis = (PRECISION2*) calloc(NUM_VISIBILITIES, sizeof(Complex));
    // host->visibilities = (Complex*) calloc(num_vis, sizeof(Complex));

    // if(*vis_uvw_coords == NULL || *measured_vis == NULL || *visibilities  == NULL)
    if(vis_uvw_coords == NULL || measured_vis == NULL)
    {
        printf("ERROR >> Unable to allocate memory for visibility information...\n\n");
        fclose(vis_file);
        exit(EXIT_FAILURE);
        // return false;
    }
    
    // Load visibility uvw coordinates into memory
    double vis_u = 0.0;
    double vis_v = 0.0;
    double vis_w = 0.0;
    double vis_real = 0.0;
    double vis_imag = 0.0;
    double vis_weight = 0.0;
    double meters_to_wavelengths = config->frequency_hz / SPEED_OF_LIGHT;

    // printf("meters_to_wavelengths %lf\n", meters_to_wavelengths);

    for(int vis_index = 0; vis_index < NUM_VISIBILITIES; ++vis_index)
    {

// #if SINGLE_PRECISION
//             fscanf(vis_file, "%f %f %f %f %f %f\n", &vis_u, &vis_v,
//                 &vis_w, &vis_real, &vis_imag, &vis_weight);
// #else
            fscanf(vis_file, "%lf %lf %lf %lf %lf %lf\n", &vis_u, &vis_v,
                &vis_w, &vis_real, &vis_imag, &vis_weight);
//#endif  

        if(config->right_ascension)  
        {   
            vis_u *= -1.0;
            vis_w *= -1.0;
        }

        if(!config->force_weight_to_one) 
        {
            vis_real *= vis_weight;
            vis_imag *= vis_weight;
        }   

        //account for w values larger than max, as we don't have kernels to deal with this
		if(ABS(vis_w) > config->max_w){
			vis_w = vis_w / ABS(vis_w) * config->max_w;
		}

        // *vis_uvw_coords[vis_index] = (Visibility) {
        //     .u = (PRECISION)(vis_u * meters_to_wavelengths),
        //     .v = (PRECISION)(vis_v * meters_to_wavelengths),
        //     .w = (PRECISION)(vis_w * meters_to_wavelengths) 
        // };
        vis_uvw_coords[vis_index].x = (vis_u * meters_to_wavelengths);
        vis_uvw_coords[vis_index].y = (vis_v * meters_to_wavelengths);

        vis_uvw_coords[vis_index].z = (vis_w * meters_to_wavelengths);

		// *measured_vis[vis_index] = (Complex) {
		// 	.real = (PRECISION)vis_real, 
		// 	.imaginary = (PRECISION)vis_imag
		// };
        measured_vis[vis_index].x = vis_real;
        measured_vis[vis_index].y = vis_imag;

        // if (vis_index < 10) printf("vis_uvw_coords[%d] = %lf %lf %lf\n", vis_index, vis_uvw_coords[vis_index].x, vis_uvw_coords[vis_index].y, vis_uvw_coords[vis_index].z);

    }
    printf("UPDATE >>> Successfully loaded %d visibilities from file...\n\n", NUM_VISIBILITIES);
    // Clean up
    fclose(vis_file);

#ifdef VERBOSE_MD5
	printf("vis_uvw_coords setup MD5 \t: ");
	MD5_Update(sizeof(PRECISION3) * NUM_VISIBILITIES, vis_uvw_coords);
#endif

    //return true;
}

void kernel_host_set_up(int NUM_KERNELS, __attribute__((unused)) int TOTAL_KERNEL_SAMPLES, Config *config, int2 *kernel_supports, PRECISION2 *kernels)
{
	printf("UPDATE >>> Loading kernel support file from %s...\n\n",config->kernel_support_file);

	FILE *kernel_support_file = fopen(config->kernel_support_file,"r");

	if (kernel_support_file == NULL) {
		printf("ERROR: Unable to open kernel support file %s\n", config->kernel_support_file);
		exit(EXIT_FAILURE);
	}
	
	int total_kernel_samples = 0;
	
	for(int plane_num = 0; plane_num < NUM_KERNELS; ++plane_num)
	{
		fscanf(kernel_support_file,"%d\n",&(kernel_supports[plane_num].x));
		kernel_supports[plane_num].y = total_kernel_samples;
		total_kernel_samples += (int)pow((kernel_supports[plane_num].x + 1) * config->oversampling, 2.0);
	}
	
	fclose(kernel_support_file);
	
	printf("UPDATE >>> Total number of samples needed to store kernels is %d...\n\n", total_kernel_samples);

	//printf("kernel_supports MD5 \t: ");
	//MD5_Update(sizeof(int2) * NUM_KERNELS, kernel_supports);

	printf("UPDATE >>> Loading kernel files file from %s real and %s imaginary...\n\n",
		config->kernel_real_file, config->kernel_imag_file);

	//now load kernels into CPU memory
	FILE *kernel_real_file = fopen(config->kernel_real_file, "r");

	if (kernel_real_file == NULL) {
		printf("ERROR: Unable to open real kernel file %s\n", config->kernel_real_file);
		exit(EXIT_FAILURE);
	}

	FILE *kernel_imag_file = fopen(config->kernel_imag_file, "r");
	if (kernel_imag_file == NULL) {
		printf("ERROR: Unable to open imaginary kernel file %s\n", config->kernel_imag_file);
		exit(EXIT_FAILURE);
	}


	int kernel_index = 0;
	for(int plane_num = 0; plane_num < NUM_KERNELS; ++plane_num)
	{
		int number_samples_in_kernel = (int) pow((kernel_supports[plane_num].x + 1) * config->oversampling, 2.0);

		for(int sample_number = 0; sample_number < number_samples_in_kernel; ++sample_number)
		{	
			double real = 0.0;
			double imag = 0.0; 

			fscanf(kernel_real_file, "%lf ", &real);
			fscanf(kernel_imag_file, "%lf ", &imag);

			kernels[kernel_index].x = real;
			kernels[kernel_index].y = imag;

			kernel_index++;
		}
	}

	fclose(kernel_real_file);
	fclose(kernel_imag_file);

}

//basically the same function as kernel_host_set_up, just loads the stipulated degridding kernels and supports instead
void degridding_kernel_host_set_up(int NUM_KERNELS, int TOTAL_KERNEL_SAMPLES, IN Config *config,
		OUT int2 *degridding_kernel_supports, OUT PRECISION2 *degridding_kernels){
	printf("UPDATE >>> Loading degridding kernel support file from %s...\n\n",config->degridding_kernel_support_file);

	FILE *kernel_support_file = fopen(config->degridding_kernel_support_file,"r");

	if(kernel_support_file == NULL)
		exit(EXIT_FAILURE);

	int total_kernel_samples = 0;

	for(int plane_num = 0; plane_num < NUM_KERNELS; ++plane_num)
	{
		fscanf(kernel_support_file,"%d\n",&(degridding_kernel_supports[plane_num].x));
		degridding_kernel_supports[plane_num].y = total_kernel_samples;
		total_kernel_samples += (int)pow((degridding_kernel_supports[plane_num].x + 1) * config->oversampling, 2.0);
	}

	fclose(kernel_support_file);

	printf("UPDATE >>> Total number of samples needed to store degridding kernels is %d...\n\n", total_kernel_samples);

	printf("kernel_supports MD5 \t: ");
	MD5_Update(sizeof(int2) * NUM_KERNELS, degridding_kernel_supports);

	printf("UPDATE >>> Loading kernel files file from %s real and %s imaginary...\n\n",
		config->degridding_kernel_real_file, config->degridding_kernel_imag_file);

	//now load kernels into CPU memory
	FILE *kernel_real_file = fopen(config->degridding_kernel_real_file, "r");
	FILE *kernel_imag_file = fopen(config->degridding_kernel_imag_file, "r");

	if(!kernel_real_file || !kernel_imag_file)
	{
		if(kernel_real_file) fclose(kernel_real_file);
		if(kernel_imag_file) fclose(kernel_imag_file);
		exit(EXIT_FAILURE);
	}

	int kernel_index = 0;
	for(int plane_num = 0; plane_num < NUM_KERNELS; ++plane_num)
	{
		int number_samples_in_kernel = (int) pow((degridding_kernel_supports[plane_num].x + 1) * config->oversampling, 2.0);

		for(int sample_number = 0; sample_number < number_samples_in_kernel; ++sample_number)
		{
			double real = 0.0;
			double imag = 0.0;

			fscanf(kernel_real_file, "%lf ", &real);
			fscanf(kernel_imag_file, "%lf ", &imag);

			degridding_kernels[kernel_index].x = real;
			degridding_kernels[kernel_index].y = imag;

			kernel_index++;
		}
	}

	fclose(kernel_real_file);
	fclose(kernel_imag_file);
}

void correction_set_up(int GRID_SIZE, PRECISION *prolate)
{
	// Allocate memory for half prolate spheroidal
	// host->prolate = (PRECISION*) calloc(config->grid_size / 2, sizeof(PRECISION));
	if(prolate == NULL)
		exit(EXIT_FAILURE);
		// return false;

	// Calculate prolate spheroidal
	create_1D_half_prolate(prolate, GRID_SIZE);

	// return true;
}

void create_1D_half_prolate(PRECISION *prolate, int grid_size)
{
	int grid_half_size = grid_size / 2;

	for(int index = 0; index < grid_half_size; ++index)
	{
		double nu = (double) index / (double) grid_half_size;
		prolate[index] = (PRECISION) prolate_spheroidal(nu);
	}
}

// Calculates a sample on across a prolate spheroidal
// Note: this is the Fred Schwabb approximation technique
double prolate_spheroidal(double nu)
{
    static double p[] = {0.08203343, -0.3644705, 0.627866, -0.5335581, 0.2312756,
        0.004028559, -0.03697768, 0.1021332, -0.1201436, 0.06412774};
    static double q[] = {1.0, 0.8212018, 0.2078043,
        1.0, 0.9599102, 0.2918724};

    int part = 0;
    int sp = 0;
    int sq = 0;
    double nuend = 0.0;
    double delta = 0.0;
    double top = 0.0;
    double bottom = 0.0;

    if(nu >= 0.0 && nu < 0.75)
    {
        part = 0;
        nuend = 0.75;
    }
    else if(nu >= 0.75 && nu < 1.0)
    {
        part = 1;
        nuend = 1.0;
    }
    else
        return 0.0;

    delta = nu * nu - nuend * nuend;
    sp = part * 5;
    sq = part * 3;
    top = p[sp];
    bottom = q[sq];

    for(int i = 1; i < 5; i++)
        top += p[sp+i] * pow(delta, i);
    for(int i = 1; i < 3; i++)
        bottom += q[sq+i] * pow(delta, i);
    return (bottom == 0.0) ? 0.0 : top/bottom;
}

void psf_host_set_up(int GRID_SIZE, int PSF_GRID_SIZE, Config *config, PRECISION *psf, double *psf_max_value)
{
	int full_psf_size_square = PSF_GRID_SIZE * PSF_GRID_SIZE;
	int psf_size_square = GRID_SIZE * GRID_SIZE;
    // host->h_psf = (PRECISION*) calloc(psf_size_square, sizeof(PRECISION));
	PRECISION* full_psf = (PRECISION*)malloc(sizeof(PRECISION) * full_psf_size_square);
	memset(full_psf, 0, sizeof(PRECISION) * full_psf_size_square);
	memset(psf, 0, sizeof(PRECISION) * psf_size_square);
	
	if(psf == NULL)
		exit(EXIT_FAILURE);
		//return false;

	FILE *psf_file = fopen(config->psf_input_file, "r");
	
	if(psf_file == NULL)
	{
		perror("PSF: ");
		exit(EXIT_FAILURE);
		//return false;
	}

	double psf_sample = 0.0;

	for(int i = 0; i < full_psf_size_square; i++)
	{

		// #if SINGLE_PRECISION
		// 	fscanf(psf_file, "%f ", &psf_sample);
		// #else
			fscanf(psf_file, "%lf ", &psf_sample);
		//#endif

		full_psf[i] = (PRECISION) psf_sample;
		//determine the max PSF sample to use for scaling the PSF and grid
		if(psf_sample > *psf_max_value)
			*psf_max_value = (PRECISION) psf_sample;

	}

	int smaller_grid_size = min(PSF_GRID_SIZE, GRID_SIZE);
	int psf_offset = max(0, (PSF_GRID_SIZE - GRID_SIZE) / 2);
	int grid_offset = max(0, (GRID_SIZE - PSF_GRID_SIZE) / 2);

	for(int y = 0; y < smaller_grid_size; ++y){
		for(int x = 0; x < smaller_grid_size; ++x){
			int psf_idx = (y + psf_offset) * PSF_GRID_SIZE + (x + psf_offset);
			int grid_idx = (y + grid_offset) * GRID_SIZE + (x + grid_offset);
			psf[grid_idx] = full_psf[psf_idx];
		}
	}

	free(full_psf);


	//scale psf
	for(int i = 0; i < psf_size_square; i++)
	{	psf[i] /= *psf_max_value;
	}
	fclose(psf_file);
	printf("UPDATE >>> PSF read from file with max value to scale grid = %f...\n\n", *psf_max_value);
	// return true;
}

void config_struct_set_up_sequel(IN Config *config_in, IN double *psf_max_value, OUT Config *config_out)
{
	memcpy(config_out, config_in, sizeof(Config));
	config_out->psf_max_value = *psf_max_value;
}

float gaussian2D(float x, float y, float x_sigma, float y_sigma)
{
	return exp(-(x * x) / (2 * x_sigma * x_sigma) - (y * y) / (2 * y_sigma * y_sigma)) / (2 * PI * x_sigma * y_sigma);
}

//Creates a CLEAN psf by locating the central lobe. This is done by traversing the dirty PSF in four directions (up, down, left, right) 
//and terminating once the second root of the derivative is found, giving us a box
void clean_psf_host_set_up(int GRID_SIZE, int GAUSSIAN_CLEAN_PSF, IN Config *config, IN PRECISION *dirty_psf, OUT PRECISION *clean_psf, OUT int2* partial_psf_halfdims){
	if(dirty_psf == NULL || clean_psf == NULL){
		exit(EXIT_FAILURE);
	}

	int center_box[4] = {GRID_SIZE / 2};
	int2 directions[4] = {{.x = 1, .y = 0}, {.x = 0, .y = 1}, {.x = -1, .y = 0}, {.x = 0, .y = -1}};

	int half_extent = GRID_SIZE / 2;
	int2 center = {.x = GRID_SIZE / 2 + 1, .y = GRID_SIZE / 2 + 1};

	for(int i = 0; i < 4; ++i){
		int prev_sign = -1;
		for(int j = 1; j < half_extent; ++j){
			int2 curr_pos = {.x = center.x + directions[i].x * j, .y = center.y + directions[i].y * j};
			int curr_pos_flat = curr_pos.x + curr_pos.y * GRID_SIZE;
			int2 last_pos = {.x = center.x + directions[i].x * (j - 1), .y = center.y + directions[i].y * (j - 1)};
			int last_pos_flat = last_pos.x + last_pos.y * GRID_SIZE;

			float deriv = dirty_psf[curr_pos_flat] - dirty_psf[last_pos_flat];
			int sign = deriv >= 0.f ? 1 : -1;

			if(sign != prev_sign){
				center_box[i] = j;
				break;
			}
		}
	}

	//just set partial psf to the largest dim of the central lobe for now
	(*partial_psf_halfdims).x = (*partial_psf_halfdims).y = MAX(MAX(MAX(center_box[0], center_box[1]), center_box[2]), center_box[3]);

	memset(clean_psf, 0, GRID_SIZE * GRID_SIZE * sizeof(PRECISION));

	//this piece of code just copies the psf defined in the box to the output. I found this to be too ugly so am looking to fit a gaussian instead
	/*for(int i = center.y - partial_psf_halfdims->y; i < center.y + partial_psf_halfdims->y; ++i){
		int idx = center.x - partial_psf_halfdims->x + i * GRID_SIZE;
		memcpy(clean_psf + idx, dirty_psf + idx, partial_psf_halfdims->x * 2 * sizeof(PRECISION));
	}*/

	//put a gaussian kernel (squared for more sharpness) over the box as using the psf directly is too ugly
	float psf_max_val = config->psf_max_value;

	float x_sigma = partial_psf_halfdims->x / 2.f;
	float y_sigma = partial_psf_halfdims->y / 2.f;

	float sum = 0.f;

	for(int y = -partial_psf_halfdims->y; y < partial_psf_halfdims->y; ++y){
		for(int x = -partial_psf_halfdims->x; x < partial_psf_halfdims->x; ++x){
			int idx = center.x + x + (center.y + y) * GRID_SIZE;
			clean_psf[idx] = gaussian2D(x, y, x_sigma, y_sigma);
			clean_psf[idx] *= clean_psf[idx];
			sum += clean_psf[idx];
			//psf_max_val = MAX(psf_max_val, dirty_psf[idx]);
		}
	}

	//normalize by max of dirty psf
	for(int y = -partial_psf_halfdims->y; y < partial_psf_halfdims->y; ++y){
		for(int x = -partial_psf_halfdims->x; x < partial_psf_halfdims->x; ++x){
			int idx = center.x + x + (center.y + y) * GRID_SIZE;
			clean_psf[idx] = clean_psf[idx] / sum * psf_max_val;
		}
	}
}

void model_set_up(int GRID_SIZE, IN Config *config, OUT PRECISION *initial_image){
	if(initial_image == NULL){
		exit(EXIT_FAILURE);
	}

	memset(initial_image, 0, GRID_SIZE * GRID_SIZE * sizeof(PRECISION));
}

void sources_set_up(int MAX_SOURCES, OUT int *num_sources, OUT PRECISION3 *sources){
	if(num_sources == NULL || sources == NULL){
		exit(EXIT_FAILURE);
	}

	*num_sources = 0;
	memset(sources, 0, MAX_SOURCES * sizeof(PRECISION3));
}

void initsink(Config* config_delta, Config* config_psi, Config* config_save, PRECISION* prolate, PRECISION* psf_delta, PRECISION* psf_psi, PRECISION* psf_save,
				PRECISION* psf_clean, int2* receiver_pairs, PRECISION2* gains, int2* kernel_supports, PRECISION2* kernels, PRECISION2* measured_vis,
				PRECISION3* vis_uwv_coords, int2* partial_psf_halfdims, int* cycle
				){
	return;
}

void initdeltasink(int GRID_SIZE, Config* config_psi, Config* config_save, PRECISION* psf_psi, PRECISION* psf_save, PRECISION* clean_psf, int2* partial_psf_halfdims
		, PRECISION* delta_image, PRECISION* image_out
		){
	return;
}

void init_sourcelist(int MAX_SOURCES, int* num_sources, PRECISION3* source_list){
	return;
}

void init_image(int GRID_SIZE, PRECISION* image){
	return;
}

void delta_pseudo(int NUM_VISIBILITIES, int NUM_RECEIVERS, int GRID_SIZE, int NUM_KERNELS, int TOTAL_KERNEL_SAMPLES, int NUM_BASELINES, int MAX_SOURCES,
				Config* config, int* cycle, PRECISION2* delta_vis, PRECISION2* gains, PRECISION* image_estimate, int2* kernel_supports,
				PRECISION2* kernels, int* num_sources_in, PRECISION* prolate, PRECISION* psf, int2* receiver_pairs, PRECISION3* source_list, PRECISION3* vis_coords,
				PRECISION* delta_image, PRECISION2* delta_vis_out, PRECISION2* gains_out, PRECISION* image_out
			){
	return;
}

void extra_delta_sinks(int NUM_VISIBILITIES, int NUM_RECEIVERS, PRECISION2* delta_vis_out, PRECISION2* gains){
	return;
}

void cycle_num(OUT int* cycle){
	return;
}

void additional_sinks(PRECISION3* source_list, int* num_sources, PRECISION* image_sink){
	return;
}

void pseudo_psi(Config* config, int* cycle, PRECISION* delta_image, PRECISION* input_model, int2* partial_psf_halfdims, PRECISION* psf,
			PRECISION* image_estimate, int* num_sources_out, PRECISION3* source_list){
	return;
}

void visibility_sink(PRECISION2* delta_vis){
	return;
}
