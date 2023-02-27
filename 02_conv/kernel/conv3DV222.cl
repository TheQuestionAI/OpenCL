// initial global_work_size = {Wout, Hout, Dout*Cout}
//#define MD 2
//#define MH 2
//#define MW 2
//#define ALIGN(SRC, BASE) (SRC + BASE - 1) / BASE * BASE 

__kernel void conv3DV222(
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
    // 1 workgroup  --->  one (MW*WGX) x (MH*WGY) x (MDC*WGZ) conv3D output-pixel block in one group output channel.
    // 1 workitem   --->  (MW * MH * MDC) Conv3D output pixels in one group output channel, with output pixel location varies with stepsize = (WGX, WGY, WGZ)
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    // We use:
    //          local_work_size[3]          = (X, Y, Z) -> {WGX, WGY, WGZ}. 
    //          global_work_size[3]         = (X, Y, Z) -> {Dout*Cgout / MDC, Wout / MW, Hout / MH}, 
    //          aligned_global_work_size[3] = (X, Y, Z) -> {align(Cgout, WGZ) * align(Dout, MDC) / MDC, align(align(Hout, MH) / MH, WGY), align(align(Wout, MW) / MW, WGX)}
    // Get the starting output-pixel location (ox, oy, oz, oc) for the current workitem from the host-output-video.
    // int os = workgroup_id * workgroup_size * MS + workitem_local_id;
    int ox  = mad24(get_group_id(2), mul24(get_local_size(2), MW ), get_local_id(2));           // ox  -> [0, Wout - 1]
    int oy  = mad24(get_group_id(1), mul24(get_local_size(1), MH ), get_local_id(1));           // oy  -> [0, Hout - 1]
    int ocz = mad24(get_group_id(0), mul24(get_local_size(0), MDC), get_local_id(0));           // ocz -> [0, Dout*Cgout - 1]

    // boundary check.
    if(Wout <= ox  || Hout <= oy || mul24(Dout, Cgout) <= ocz) return;

    // Figure out the valid output-pixels of the (MW x MH x MDC) that the current workitem will work on.
    // example: Wout = 19           -> the index of last element is idx = 18. 
    //          gx = 5, MW = 3,     -> ox = gx * MW = 15,   -> the starting output-pixel X index is ox = 15.
    //          WGX = 4             -> the valid output X index location is ox = 15. Only 1 valid location.
    //                              -> Mw = min(MW, 1 + (Wout - ox) / WGX) = min(4, 1 + (22 - 15) / 5) = min(4, 1 + 1) = 2.
    int Mw = min(MW, 1 + (Wout - ox - 1) / WGX);        // ceiling  = (x + y - 1) / y = 1 + (x - 1) / y;    The last one can avoid overflow.
    int Mh = min(MH, 1 + (Hout - oy - 1) / WGY);
    int Mdc = min(MDC, 1 + (mad24(Dout, Cgout, -ocz) - 1) / WGZ);

    // Calculate the Conv3D current output location (cox, coy, coz, coc) -> (w_out, h_out, d_out, c_out) from host-output-video.
    int curOx = NULL;           // varies in for loop.
    int curOy = NULL;           // varies in for loop.
    int curOcz = NULL;          // varies in for loop.
    int curOc = NULL;           // varies in for loop.
    int curOz = NULL;           // varies in for loop.

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

    // Figure out the Conv3D current input pixel (imgCurX, imgCurY, imgCurZ) in image2darray-input, where we obtain it to multiply corresponding filter value for conv3D operation.
    int img2dArrCurrentIx = NULL;   // img2dArrCurrentIx varies by different Cgin/Dk/Hk/Wk and img2dFy. We determine it in for loop.
    int img2dArrCurrentIy = NULL;   // img2dArrCurrentIy varies by different Cgin/Dk/Hk/Wk and img2dFy. We determine it in for loop.
    int img2dArrCurrentIz = NULL;   // img2dArrCurrentIz varies by different Cgin/Dk/Hk/Wk and img2dFy. We determine it in for loop.

    // a) Figure out the target filter in image2d-filters. According to the layout we select, every column in image2d-filters corresponds to a filter.
    // b) Figure out the starting coordinate (imgX, imgY) of current group input channel of target filter in image2d-filters, where we obtain the target filter group input channel for conv3D operation. 
    // filters image2d: H x W -> Cgin*Dk*Hk*Wk x Cout*4cin          <---        every column in filters-image2d corresponds to a filter.
    //int img2dFx = NULL;     // img2dFx (img2dFx in [0, Cout - 1]) will vary by different filter selected. We determine it in for loop.
    int img2dFy = NULL;       // img2dFy varies by different Cgin. We determine it in for loop.
    
    // Figure out the current group filter value with respect to current group input channel values of target filter in image2d-filters.
    int img2dCurrentFy = NULL;  // img2dCurrentFy varies by different Cgin/Dk/Hk/Wk and img2dFy. We determine it in for loop.

    // Figure out the target bias in image1d-biases. According to the 1D image layout we select, there are Cgout group biases.
    // int img1dBx = NULL;     // img1dBx (img1dBx in [0, Cgout - 1]) varies by different group filter selected. We determine it in for loop. 

    // pre-cacculate filter size and filter 2D slice size which may save some compuation in for loops.
    int filterSize = mul24(mul24(Wk, Hk), Dk);      
    int fliterSliceSize = mul24(Wk, Hk);

    // Initialize the 4 output points in 1 output pixel.
    float4 conv3DResults;
    float4 conv3DGroupLevelResults[4];

    // Initialize some temporary variables.
    float4 inputPatchValue;

    // loop all valid output-pixels for each workitem, stepsize = (WGX, WGY, WGZ).
    for(int dc = 0; dc < Mdc; ++dc) {
        curOcz = mad24(dc, WGZ, ocz);              // Calculate the Conv3D current output location (cox, coy, coz, coc) -> (w_out, h_out, d_out, c_out) from host-output-video.
        curOc = curOcz / Dout;
        curOz = mad24(-curOc, Dout, curOcz);
        iz = mad24(curOz, Sz, -Pz);               // Calculate the Conv3D current starting input location (ix, iy, iz) -> (d_in, h_in, w_in) of the input patch from host-input-video.
        for(int h = 0; h < Mh; ++h) {
            curOy = mad24(h, WGY, oy);            // Calculate the Conv3D current output location (cox, coy, coz, coc) -> (w_out, h_out, d_out, c_out) from host-output-video.
            iy = mad24(curOy, Sy, -Py);           // Calculate the Conv3D current starting input location (ix, iy, iz) -> (d_in, h_in, w_in) of the input patch from host-input-video.
            for(int w = 0; w < Mw; ++w) {
                curOx = mad24(w, WGX, ox);        // Calculate the Conv3D current output location (cox, coy, coz, coc) -> (w_out, h_out, d_out, c_out) from host-output-video.
                ix = mad24(curOx, Sx, -Px);       // Calculate the Conv3D current starting input location (ix, iy, iz) -> (d_in, h_in, w_in) of the input patch from host-input-video.

                // Initialize the 4 output points in 1 output pixel.
                conv3DResults = (float4)(0.0f, 0.0f, 0.0f, 0.0f);      
                
                // Figure out the output Z coordinate in image2darray-output, img2darrOz varies by differnet output channel. 
                // output image2darray: Cgout*Dout x Hout x Wout*4cout
                img2dArrOz = mad24(curOc, Dout, curOz);   // First select the group output channel (group channel strided by Dout distance), then from the group output channel selects the target output Z coordinate.      

                // 2-step conv3D for 1 output pixel (4 output points): 1. group_input_channel level conv3D; 2. conv3D_result_point level add up.
                // group_input_channel level conv3D, each output point needs a vector4 to store conv3D results.
                conv3DGroupLevelResults[0] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                conv3DGroupLevelResults[1] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                conv3DGroupLevelResults[2] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                conv3DGroupLevelResults[3] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

                // Loop all group input channels of the input patch for conv3D operation.
                for(int cig = 0; cig < Cgin; ++cig) {
                    // img2darrIz varies by differnet group input channel, we determine the starting coordinate img2dArrIz for current group input channel in input image2darray in for loop.
                    // input image2dArray: Cgin*Din x Hin x Win*4cin
                    img2dArrIz = mad24(cig, Din, iz);   // Each group input channel is seperated by Din distance, so we stride by Din to determine the current group input channel.  
                                                        // Then we determine the starting input Z coordinate of the current group input channel in image2darray-input.
                    
                    // Figure out the starting coordinate (imgX, imgY) of current group input channel of target filter in image2d-filters.
                    // filters image2d: Cgin*Dk*Hk*Wk x Cout*4cin
                    img2dFy = mul24(cig, filterSize);     // Each group input filter channel is seperated by Dk*Hk*Wk distance, so we stride by Dk*Hk*Wk to determine the current group input filter channel. 
                    // filterSize = Dk * Hk * Wk
                    
                    // loop all points in selected input channel of filter and of input patch for conv3d operation. filter_shape = (Dk, Hk, Wk)
                    for(int kz = 0; kz < Dk; ++kz) {
                        img2dArrCurrentIz = mad24(kz, Lz, img2dArrIz);
                        for(int ky = 0; ky < Hk; ++ky) {
                            img2dArrCurrentIy = mad24(ky, Ly, iy);
                            for(int kx = 0; kx < Wk; ++kx) {
                                img2dArrCurrentIx = mad24(kx, Lx, ix);
                                img2dCurrentFy = mad24(kz, fliterSliceSize, img2dFy) + mad24(ky, Wk, kx);

                                inputPatchValue = ((unsigned)mad24(-cig, Din, img2dArrCurrentIz) < Din) * read_imagef(input, smp, (int4)(img2dArrCurrentIx, img2dArrCurrentIy, img2dArrCurrentIz, 0));
                                
                                // perform conv3d operation between the same target input patch but different target filter for each output point.
                                // loop the 4 output points inside the target output pixel to obtain the conv3D operation results for each.
                                // img2dFx = mad24(oc, 4, idx); idx = [0, 1, 2, 3]. Pick up the target filter for conv3d in this run.
                                conv3DGroupLevelResults[0] += inputPatchValue * read_imagef(filters, smp, (int2)(mad24(curOc, 4, 0), img2dCurrentFy));
                                conv3DGroupLevelResults[1] += inputPatchValue * read_imagef(filters, smp, (int2)(mad24(curOc, 4, 1), img2dCurrentFy));
                                conv3DGroupLevelResults[2] += inputPatchValue * read_imagef(filters, smp, (int2)(mad24(curOc, 4, 2), img2dCurrentFy));
                                conv3DGroupLevelResults[3] += inputPatchValue * read_imagef(filters, smp, (int2)(mad24(curOc, 4, 3), img2dCurrentFy));
                            }
                        }              
                    }
                }

                // sum up all group channel conv results to get the single output point convolution results.
                conv3DResults += (float4)(conv3DGroupLevelResults[0].x, conv3DGroupLevelResults[1].x, conv3DGroupLevelResults[2].x, conv3DGroupLevelResults[3].x);
                conv3DResults += (float4)(conv3DGroupLevelResults[0].y, conv3DGroupLevelResults[1].y, conv3DGroupLevelResults[2].y, conv3DGroupLevelResults[3].y);
                conv3DResults += (float4)(conv3DGroupLevelResults[0].z, conv3DGroupLevelResults[1].z, conv3DGroupLevelResults[2].z, conv3DGroupLevelResults[3].z);
                conv3DResults += (float4)(conv3DGroupLevelResults[0].w, conv3DGroupLevelResults[1].w, conv3DGroupLevelResults[2].w, conv3DGroupLevelResults[3].w);
                // read biases and add it to the conv3DResults.
                conv3DResults += read_imagef(biases, smp, curOc);
                
                write_imagef(output, (int4)(curOx, curOy, img2dArrOz, 0), conv3DResults);    
            }
        }
    }
}