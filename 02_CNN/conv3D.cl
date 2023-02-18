
__kernel void conv3D(
    __read_only image2d_array_t input,              // image2dArray: Cgin*Din x Hin x Win*4cin
    __read_only image2d_t filters,                  // image2d     : Cgin*Dk*Hk*Wk x Cout*4cin
    __read_only image1d_t biases,                   // image1d     : Cgout*4cout
    __write_only image2d_array_t output,            // image2dArray: Cgout*Dout x Hout x Wout*4cout
    int Din,                // host-input depth
    int Cgin,               // host-input group channel
    int Wout,               // host-output width
    int Hout,               // host-output height
    int Dout,               // host-output depth
    int Cgout,              // host-output group channel = numFilters / 4.
    int Wk,                 // filter width
    int Hk,                 // filter height
    int Dk,                 // filter depth
    int Sx,                 // strideX
    int Sy,                 // strideY
    int Sz,                 // strideZ
    int Px,                 // PaddingX
    int Py,                 // PaddingY
    int Pz,                 // PaddingZ
    int Lx,                 // DilationX
    int Ly,                 // DilationY
    int Lz                  // Dilationz
    )
{
    const sampler_t smp1 = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    const sampler_t smp2 = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    const sampler_t smp3 = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    // Each workitem will work on Cout conv3D output points (Cgout = Cout/4 Conv3D output pixels), one output point per one output channel (one output pixel per one group output channel). 
    // To be more clear, 1. The output points that current workitem works on, are with exactly the same (d_out, h_out, w_out) coordinates but different in c_out (output channel) coordinate.
    //                   2. Also, the input patch ([d_in:d_in+fx], [h_in:h_in+fy], [w_in:w_in+fy], [0:Cin]) used for conv3D operation is exactly the same for all output points.
    //                   3. What that said is, Cout output points are calculated using the exactly same input patch convoluted with just different 3D filter.

    // We use global_work_size[3] -> (X, Y, Z) -> {Wout, Hout, Dout}, local_work_size[3] = {1, 1, 1}. 
    // Get the output location (ox, oy, oz) -> (d_out, h_out, w_out) for output points that the current workitem works on from the host-output-video.
    // Remember, the output points the current workitem works on, are with exactly the same (ox, oy, oz) -> (d_out, h_out, w_out) coordinates but different in c_out (output channel) coordinate.
    int ox = get_global_id(0);      // ox -> [0, Wout - 1]
    int oy = get_global_id(1);      // oy -> [0, Hout - 1]
    int oz = get_global_id(2);      // oz -> [0, Dout - 1]

    // boundary check.
    if(ox >= Wout || oy >= Hout || oz >= Dout) return;

    // Calculate the Conv3D starting input location (ix, iy, iz) -> (d_in, h_in, w_in) of the input patch from host-input-video.
    // Remember, the input patch ([ix:ix+fx], [iy:iy+fy], [iz:iz+fz], [0:Cin]) used for conv3D operation is exactly the same for all output points that the current workitem works on.
    int ix = ox * Sx - Px;       // Here we only need p_left, that is we alwasy assume (p_left + 1) = p_right and p_left + p_right = 2*Pvalue.
    int iy = oy * Sy - Py;       // Thus, we take the floor(Pvalue) to get p_left. p_left = floor(Pvalue) = int(Pvalue)
    int iz = oz * Sz - Pz;       

    // Figure out the output coordinate (imgX, imgY, imgZ) in image2darray-output, where we stores the conv3D operation results. 
    // i.e, We need the map the host-output-video output location (ox, oy, oz, oc) to output coordinate (imgX, imgY, imgZ) in image2darray-output. 
    // output image2darray: Cgout*Dout x Hout x Wout*4cout
    int img2dArrOz = NULL;      // Each work-item works on Cgout output pixels for Cgout group output channels, thus img2darrOz varies by differnet group output channel. We dertermine the output Z coordinate in for loop.
    // int img2dArrOx = ox;     // ouput X coordinate in image2darray-output is exactly the same of output X location in host-output.
    // int img2dArrOy = oy;     // ouput Y coordinate in image2darray-output is exactly the same of output Y location in host-output.
   
    // Figure out the Conv3D starting input coordinate (imgX, imgY, imgZ) in image2darray-input, where we obtain each input channel of input patch.
    // i.e, We need the map the host-input-patch starting input location (ix, iy, iz, ic) to starting input coordinate (imgX, imgY, imgZ) in image2darray-input. 
    // input image2darray: Cgin*Din x Hin x Win*4cin
    int img2dArrIz = NULL;      // Conv3D opertions needs to loop all group input channels, thus img2darrIz varies by differnet group input channel, we determine the input starting Z coordinate for each group input channel in for loop.
    // int img2dArrIx = ix;     // input starting X coordinate in image2darray is exactly the same of input starting X location in host-input.
    // int img2dArrIy = iy;     // input starting Y coordinate in image2darray is exactly the same of input starting Y location in host-input.

    // a) Figure out the target filter in image2d-filters. According to the layout we select, every column in image2d-filters corresponds to a filter.
    // b) Figure out the starting coordinate (imgX, imgY) of current group input channel of target filter in image2d-filters, where we obtain the target filter group input channel for conv3D operation. 
    // filters image2d: H x W -> Cgin*Dk*Hk*Wk x Cout*4cin          <---        every column in filters-image2d corresponds to a filter.
    int img2dFx = NULL;     // img2dFx (img2dFx in [0, Cout - 1]) will vary by different filter selected. We determine it in for loop.
    int img2dFy = NULL;     // img2dFy varies by different Cgin/Dk/Hk/Wk. We determine it in for loop.
    
    // Figure out the target bias in image1d-biases. According to the 1D image layout we select, there are Cgout group biases.
    int img1dBx = NULL;     // img1dBx (img1dBx in [0, Cgout - 1]) varies by different group filter selected. We determine it in for loop. 

    // pre-cacculate filter size and filter 2D slice size which may save some compuation in for loops.
    int filterSize = Wk * Hk * Dk;      
    int fliterSliceSize = Wk * Hk;

    // 1 work-item is responsible for Cout = Cgout*4 output points, i.e. Cgout output pixels.
    // We loop all group output channels, and each group output channel we calculate 1 output pixel (i.e. 4 output points).
    for(int cog = 0; cog < Cgout; ++cog) {
        // Initialize the 4 output points in 1 output pixel.
        float4 conv3DResults = {0.0f, 0.0f, 0.0f, 0.0f};      
        
        // Figure out the output Z coordinate in image2darray-output, img2darrOz varies by differnet output channel. 
        // output image2darray: Cgout*Dout x Hout x Wout*4cout
        img2dArrOz = cog * Dout + oz;   // First select the group output channel (group channel strided by Dout distance), then from the group output channel selects the target output Z coordinate.      

        // Figure out the bias X coordinate in image1d-biases, img1dBx (img1dBx in [0, Cgout - 1]) varies by different group filter selected.
        // img1dBx = cog;   // bias X coordinate in image1d-biases is exactly the same of the group filter selected.

        // 2-step conv3D for 1 output pixel (4 output points): 1. group_input_channel level conv3D; 2. conv3D_result_point level add up.
        // group_input_channel level conv3D, each output point needs a vector4 to store conv3D results.
        float4 conv3DGroupLevelResults[4] = {{0.0f, 0.0f, 0.0f, 0.0f},
                                             {0.0f, 0.0f, 0.0f, 0.0f},
                                             {0.0f, 0.0f, 0.0f, 0.0f},
                                             {0.0f, 0.0f, 0.0f, 0.0f}};
        // perform conv3d operation between the same target input patch but different target filter for each output point.
        for(int idx = 0; idx < 4; ++idx) {  // loop the 4 output points inside the target output pixel to obtain the conv3D operation results for each.
            // Pick up the target filter for conv3d in this run.
            // filters image2d: Cgin*Dk*Hk*Wk x Cout*4cin. i.e. every column in filters-image2d corresponds to a filter.
            img2dFx = cog * 4 + idx;    // img2dFx in [0, Cout - 1]. So we need to multiple 4 back to cog (cog = Cout / 4) to select target "group" filter then select the single target filter.      

            // Loop all group input channels of the input patch for conv3D operation.
            for(int cig = 0; cig < Cgin; ++cig) {
                // img2darrIz varies by differnet group input channel, we determine the starting coordinate img2dArrIz for current group input channel in input image2darray in for loop.
                // input image2dArray: Cgin*Din x Hin x Win*4cin
                img2dArrIz = cig * Din + iz;   // Each group input channel is seperated by Din distance, so we stride by Din to determine the current group input channel.  
                                               // Then we determine the starting input Z coordinate of the current group input channel in image2darray-input.
                
                // Figure out the starting coordinate (imgX, imgY) of current group input channel of target filter in image2d-filters.
                // filters image2d: Cgin*Dk*Hk*Wk x Cout*4cin
                img2dFy = cig * filterSize;     // Each group input filter channel is seperated by Dk*Hk*Wk distance, so we stride by Dk*Hk*Wk to determine the current group input filter channel. 
                // filterSize = Dk * Hk * Wk
                
                // loop all points in selected input channel of filter and of input patch for conv3d operation.
                // filter_shape = (Dk, Hk, Wk)
                for(int kz = 0; kz < Dk; ++kz) {
                    int img2dArrCurrentIz = img2dArrIz + kz * Lz;
                    if(img2dArrCurrentIz < cig*Din || img2dArrCurrentIz >= (cig+1)*Din) //continue;
                    {
                        img2dArrCurrentIz = -1;
                    }
                    for(int ky = 0; ky < Hk; ++ky) {
                        int img2dArrCurrentIy = iy + ky * Ly;
                        for(int kx = 0; kx < Wk; ++kx) {
                            int img2dArrCurrentIx = ix + kx * Lx;
                            int img2dCurrentFy = img2dFy + kz * fliterSliceSize + ky * Wk + kx;

                            float4 inputPatchValue = read_imagef(input, smp1, (int4)(img2dArrCurrentIx, img2dArrCurrentIy, img2dArrCurrentIz, 0));
                            
                            if((img2dArrCurrentIx != -1 && img2dArrCurrentIy != -1) && img2dArrCurrentIz == -1)
                                printf("output=(%d,%d,%d,%d) input=(%d,%d,%d) inputValue=[%f,%f,%f,%f]\n", oz, oy, ox, cog, img2dArrCurrentIz, img2dArrCurrentIy, img2dArrCurrentIx, inputPatchValue.x, inputPatchValue.y, inputPatchValue.z, inputPatchValue.w); 

                            float4 filterValue = read_imagef(filters, smp2, (int2)(img2dFx, img2dCurrentFy));
                            
                            conv3DGroupLevelResults[idx] += inputPatchValue * filterValue;
                        }
                    }
                }              
            }
        }

        // sum up all group channel conv results to get the single output point convolution results.
        conv3DResults.x = conv3DGroupLevelResults[0].x + conv3DGroupLevelResults[0].y + conv3DGroupLevelResults[0].z + conv3DGroupLevelResults[0].w;
        conv3DResults.y = conv3DGroupLevelResults[1].x + conv3DGroupLevelResults[1].y + conv3DGroupLevelResults[1].z + conv3DGroupLevelResults[1].w;
        conv3DResults.z = conv3DGroupLevelResults[2].x + conv3DGroupLevelResults[2].y + conv3DGroupLevelResults[2].z + conv3DGroupLevelResults[2].w;
        conv3DResults.w = conv3DGroupLevelResults[3].x + conv3DGroupLevelResults[3].y + conv3DGroupLevelResults[3].z + conv3DGroupLevelResults[3].w;

        // read biase and add it to the conv3DResults.
        conv3DResults += read_imagef(biases, smp3, cog);

        // write back the results to output.
        write_imagef(output, (int4)(ox, oy, img2dArrOz, 0), conv3DResults);

        //printf("output=(%d,%d,%d,%d) input=(%d,%d,%d) res=[%f,%f,%f,%f]\n", oz, oy, ox, cog, iz, iy, ix, conv3DResults.x, conv3DResults.y, conv3DResults.z, conv3DResults.w);

        //printf("[ %f, %f, %f, %f ]\n", conv3DResults.x, conv3DResults.y, conv3DResults.z, conv3DResults.w);
        //printf("[ globalx: %d, globaly: %d, globalz: %d ] completed.\n", ox, oy, oz);
    }

    //printf("[ globalx: %d, globaly: %d, globalz: %d ] completed.\n", ox, oy, oz);
}