// initial global_work_size = {Wout, Hout, Dout*Cout}
//#define MC 2
//#define MD 2
//#define MH 2
//#define MW 2

__kernel void conv3DV2(
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
    // 1 workitem  --->  (MW * MH * MD * MC * 4) conv3D output points  --->  (MW * MH * MD * MC) Conv3D output pixels  ---> (MW x MH x MD) patches in MC group channels.
    // Each workitem will work on MW*MH*MD*MC*4 conv3D output points (MW*MH*MD output points in each output channel and total MC*4 output channels, or MW*MH*MD*4 output points in each group output channel and total MC group output channels).
    // In other word, each workitem will work on MW*MH*MD*MC Conv3D output pixels, i.e. MW*MH*MD output pixel per one group output channel and total MC group output channels.
    // That means, each workitem will work on a MW x MH x MD small output-pixel-patch in each group output channel, and total MC group output channels.
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    //printf("MW = %d, MH = %d, MD = %d, MC = %d \n", MW, MH, MD, MC);

    // We use:
    //          local_work_size[3]          = (X, Y, Z) -> {WGX, WGY, WGZ}. 
    //          global_work_size[3]         = (X, Y, Z) -> {Dout*Cgout / (MD*MC), Wout / MW, Hout / MH}, 
    //          aligned_global_work_size[3] = (X, Y, Z) -> {align(align(Dout*Cgout, MD*MC) / (MD*MC), WGZ), align(align(Hout, MH) / MH, WGY), align(align(Wout, MW) / MW, WGX)}
    // Get the current workitem global location in the global_work_size.
    //int gx = get_global_id(2);
    //int gy = get_global_id(1);
    //int gz = get_global_id(0);

    // Get the starting output-pixel location (ox, oy, oz, oc) for the current workitem from the host-output-video.
    // 1 workitem  -->  (MW * MH * MD * MC) Conv3D output pixels, so recover the true work shape for host-output-video by multiply back the number of work pixels in each dimension. 
    // Remember, the output-pixel-patches (MW x MH x MD) for each workitem are with exactly same starting output location (ox, oy, oz) but different in group output channel (oc).
    int ox = get_global_id(2) * MW;                         // ox -> [0, Wout - 1]
    int oy = get_global_id(1) * MH;                         // oy -> [0, Hout - 1]
    int oc = get_global_id(0) * MD * MC / Dout;             // oc -> [0, Cgout - 1]    original_global_work_size_z = Dout*Cgout, need to separate c_gout and d_out
    int oz = get_global_id(0) * MD * MC - oc * Cgout;       // oz -> [0, Dout - 1]

    // printf("Dout = %d, Cgout = %d \n", Dout, Cgout);
    // gz * MD * MC = 1, Dout = 2, Cgout = 1. Dout * Cgout = 2

    // boundary check.
    //if(ox >= Wout || oy >= Hout || oz >= Dout || oc >= Cgout) return;

    // Figure out the valid region of the output-pixel-patches (MW x MH x MD) that the current workitem will work on.
    // example: Wout = 21           -> the index of last element is idx = 20 
    //          gx = 10, MW = 2,    -> ox = gx * MW = 20,   -> the valid region is only the last element with index = 20, there is no more distance we can go. idx = 21 is invalid.
    //                              -> Mw = min(MW, Wout - ox) = min(2, 21 - 20) = 1
    int Mw = max(0, min(MW, Wout - ox));
    int Mh = max(0, min(MH, Hout - oy));
    int Md = max(0, min(MD, Dout - oz));
    int Mc = max(0, min(MC, Cgout - oc));
    
    //printf("(ox=%d, oy=%d, oz=%d, oc=%d), (Mw=%d, Mh=%d, Md=%d, Mc=%d) \n", ox, oy, oz, oc, Mw, Mh, Md, Mc);


    // Calculate the Conv3D current output location (cox, coy, coz, coc) -> (w_out, h_out, d_out, c_out) from host-output-video.
    int curOx = NULL;       // varies in for loop
    int curOy = NULL;       // varies in for loop
    int curOz = NULL;       // varies in for loop
    int curOc = NULL;       // varies in for loop

    // Calculate the Conv3D current starting input location (ix, iy, iz) -> (d_in, h_in, w_in) of the input patch from host-input-video.
    int ix = NULL;               // depends on curOx, so varies in for loop.
    int iy = NULL;               // depends on curOy, so varies in for loop.
    int iz = NULL;               // depends on curOz, so varies in for loop.
    
    // Figure out the output coordinate (imgX, imgY, imgZ) in image2darray-output, where we stores the conv3D operation results. 
    // i.e, We need the map the host-output-video output location (ox, oy, oz, oc) to output coordinate (imgX, imgY, imgZ) in image2darray-output. 
    // output image2darray: Cgout*Dout x Hout x Wout*4cout
    int img2dArrOz = NULL;          // Each work-item works on Cgout output pixels for Cgout group output channels, thus img2darrOz varies by differnet group output channel. We dertermine the output Z coordinate in for loop.
    // int img2dArrOx = curOx;      // ouput X coordinate in image2darray-output is exactly the same of output X location in host-output.
    // int img2dArrOy = curOy;      // ouput Y coordinate in image2darray-output is exactly the same of output Y location in host-output.
   
    // Figure out the Conv3D starting input coordinate (imgX, imgY, imgZ) in image2darray-input, where we obtain each input channel of input patch.
    // i.e, We need the map the host-input-patch starting input location (ix, iy, iz, ic) to starting input coordinate (imgX, imgY, imgZ) in image2darray-input. 
    // input image2darray: Cgin*Din x Hin x Win*4cin
    int img2dArrIz = NULL;          // Conv3D opertions needs to loop all group input channels, thus img2darrIz varies by differnet group input channel, we determine the input starting Z coordinate for each group input channel in for loop.
    // int img2dArrIx = curIx;      // input starting X coordinate in image2darray is exactly the same of input starting X location in host-input.
    // int img2dArrIy = curIy;      // input starting Y coordinate in image2darray is exactly the same of input starting Y location in host-input.

    // a) Figure out the target filter in image2d-filters. According to the layout we select, every column in image2d-filters corresponds to a filter.
    // b) Figure out the starting coordinate (imgX, imgY) of current group input channel of target filter in image2d-filters, where we obtain the target filter group input channel for conv3D operation. 
    // filters image2d: H x W -> Cgin*Dk*Hk*Wk x Cout*4cin          <---        every column in filters-image2d corresponds to a filter.
    int img2dFx = NULL;     // img2dFx (img2dFx in [0, Cout - 1]) will vary by different filter selected. We determine it in for loop.
    int img2dFy = NULL;     // img2dFy varies by different Cgin/Dk/Hk/Wk. We determine it in for loop.
    
    // Figure out the target bias in image1d-biases. According to the 1D image layout we select, there are Cgout group biases.
    // int img1dBx = NULL;     // img1dBx (img1dBx in [0, Cgout - 1]) varies by different group filter selected. We determine it in for loop. 

    // pre-cacculate filter size and filter 2D slice size which may save some compuation in for loops.
    int filterSize = Wk * Hk * Dk;      
    int fliterSliceSize = Wk * Hk;

    // Initialize the 4 output points in 1 output pixel.
    float4 conv3DResults;
    float4 conv3DGroupLevelResults[4];

    for(int c = 0; c < Mc; ++c) {               // loop target group output channel, as the distance between two adjcent group output channel is far.
        curOc = oc + c;                         // Calculate the Conv3D current output location (cox, coy, coz, coc) -> (w_out, h_out, d_out, c_out) from host-output-video.
        for(int d = 0; d < Md; ++d) {
            curOz = oz + d;                     // Calculate the Conv3D current output location (cox, coy, coz, coc) -> (w_out, h_out, d_out, c_out) from host-output-video.
            iz = curOz * Sz - Pz;               // Calculate the Conv3D current starting input location (ix, iy, iz) -> (d_in, h_in, w_in) of the input patch from host-input-video.
            for(int h = 0; h < Mh; ++h) {
                curOy = oy + h;                 // Calculate the Conv3D current output location (cox, coy, coz, coc) -> (w_out, h_out, d_out, c_out) from host-output-video.
                iy = curOy * Sy - Py;           // Calculate the Conv3D current starting input location (ix, iy, iz) -> (d_in, h_in, w_in) of the input patch from host-input-video.
                for(int w = 0; w < Mw; ++w) {
                    curOx = ox + w;             // Calculate the Conv3D current output location (cox, coy, coz, coc) -> (w_out, h_out, d_out, c_out) from host-output-video.
                    ix = curOx * Sx - Px;       // Calculate the Conv3D current starting input location (ix, iy, iz) -> (d_in, h_in, w_in) of the input patch from host-input-video.

                    // Initialize the 4 output points in 1 output pixel.
                    conv3DResults = (float4)(0.0f, 0.0f, 0.0f, 0.0f);      
                    
                    // Figure out the output Z coordinate in image2darray-output, img2darrOz varies by differnet output channel. 
                    // output image2darray: Cgout*Dout x Hout x Wout*4cout
                    img2dArrOz = curOc * Dout + curOz;   // First select the group output channel (group channel strided by Dout distance), then from the group output channel selects the target output Z coordinate.      

                    // Figure out the bias X coordinate in image1d-biases, img1dBx (img1dBx in [0, Cgout - 1]) varies by different group filter selected.
                    // img1dBx = cog;   // bias X coordinate in image1d-biases is exactly the same of the group filter selected.

                    // 2-step conv3D for 1 output pixel (4 output points): 1. group_input_channel level conv3D; 2. conv3D_result_point level add up.
                    // group_input_channel level conv3D, each output point needs a vector4 to store conv3D results.
                    conv3DGroupLevelResults[0] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                    conv3DGroupLevelResults[1] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                    conv3DGroupLevelResults[2] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                    conv3DGroupLevelResults[3] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                    // perform conv3d operation between the same target input patch but different target filter for each output point.
                    for(int idx = 0; idx < 4; ++idx) {  // loop the 4 output points inside the target output pixel to obtain the conv3D operation results for each.
                        // Pick up the target filter for conv3d in this run.
                        // filters image2d: Cgin*Dk*Hk*Wk x Cout*4cin. i.e. every column in filters-image2d corresponds to a filter.
                        img2dFx = curOc * 4 + idx;    // img2dFx in [0, Cout - 1]. So we need to multiple 4 back to cog (cog = Cout / 4) to select target "group" filter then select the single target filter.      

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
                                for(int ky = 0; ky < Hk; ++ky) {
                                    int img2dArrCurrentIy = iy + ky * Ly;
                                    for(int kx = 0; kx < Wk; ++kx) {
                                        int img2dArrCurrentIx = ix + kx * Lx;
                                        int img2dCurrentFy = img2dFy + kz * fliterSliceSize + ky * Wk + kx;

                                        float4 inputPatchValue = ((unsigned)(img2dArrCurrentIz - cig*Din) < Din) * read_imagef(input, smp, (int4)(img2dArrCurrentIx, img2dArrCurrentIy, img2dArrCurrentIz, 0));
                                        float4 filterValue = read_imagef(filters, smp, (int2)(img2dFx, img2dCurrentFy));
                                        
                                        conv3DGroupLevelResults[idx] += inputPatchValue * filterValue;
                                    }
                                }
                            }              
                        }
                    }

                    // sum up all group channel conv results to get the single output point convolution results.
                    //conv3DResults += (float4)(conv3DGroupLevelResults[0].x, conv3DGroupLevelResults[1].x, conv3DGroupLevelResults[2].x, conv3DGroupLevelResults[3].x);
                    //conv3DResults += (float4)(conv3DGroupLevelResults[0].y, conv3DGroupLevelResults[1].y, conv3DGroupLevelResults[2].y, conv3DGroupLevelResults[3].y);
                    //conv3DResults += (float4)(conv3DGroupLevelResults[0].z, conv3DGroupLevelResults[1].z, conv3DGroupLevelResults[2].z, conv3DGroupLevelResults[3].z);
                    //conv3DResults += (float4)(conv3DGroupLevelResults[0].w, conv3DGroupLevelResults[1].w, conv3DGroupLevelResults[2].w, conv3DGroupLevelResults[3].w);
                    conv3DResults.x = conv3DGroupLevelResults[0].x + conv3DGroupLevelResults[0].y + conv3DGroupLevelResults[0].z + conv3DGroupLevelResults[0].w;
                    conv3DResults.y = conv3DGroupLevelResults[1].x + conv3DGroupLevelResults[1].y + conv3DGroupLevelResults[1].z + conv3DGroupLevelResults[1].w;
                    conv3DResults.z = conv3DGroupLevelResults[2].x + conv3DGroupLevelResults[2].y + conv3DGroupLevelResults[2].z + conv3DGroupLevelResults[2].w;
                    conv3DResults.w = conv3DGroupLevelResults[3].x + conv3DGroupLevelResults[3].y + conv3DGroupLevelResults[3].z + conv3DGroupLevelResults[3].w;

                    // read biase and add it to the conv3DResults.
                    conv3DResults += read_imagef(biases, smp, curOc);

                    // write back the results to output.
                    write_imagef(output, (int4)(curOx, curOy, img2dArrOz, 0), conv3DResults);

                    //if(get_global_id(0) == 0) {
                    //printf("thread=(%d,%d,%d), satrt_output=(%d,%d,%d,%d), output=(%d,%d,%d,%d) input=(%d,%d,%d) res=[%f,%f,%f,%f]\n", gx, gy, gz, ox, oy, oz, oc, curOx, curOz, curOy, curOc, ix, iy, iz, conv3DResults.x, conv3DResults.y, conv3DResults.z, conv3DResults.w);
                    //}
                    
                    //printf("[ %f, %f, %f, %f ]\n", conv3DResults.x, conv3DResults.y, conv3DResults.z, conv3DResults.w);
                    //printf("[ globalx: %d, globaly: %d, globalz: %d ] completed.\n", ox, oy, oz);

                }
            }
        }
    }
    //printf("[ globalx: %d, globaly: %d, globalz: %d ] completed.\n", ox, oy, oz);
}