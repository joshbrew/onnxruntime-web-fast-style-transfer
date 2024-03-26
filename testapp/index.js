import * as ort from 'onnxruntime-web'
import * as wgpuort from 'onnxruntime-web/webgpu'

const modelName = 'model.onnx';

let session;

const init = async () => {
    //https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/api-usage_session-options
    try{ //WebGPU
        session = await wgpuort.InferenceSession.create(location.origin+"/models/"+modelName, {
            executionProviders: ['webgpu'] //'wasm' 'webgl' 'webgpu'
        });
        console.log("Created WebGPU ONNX session");
    } catch(er) {
        console.error("WebGPU ONNX Create Session error:", er);
        try{ //WebGL fallback
            session = await ort.InferenceSession.create(location.origin+"/models/"+modelName, {
                executionProviders: ['webgl'] //'wasm' 'webgl' 'webgpu'
            });
            console.log("Created WebGL ONNX session");
        } catch(er) {
            console.error("WebGL ONNX Create Session error:", er);
            try{ //CPU fallback
                session = await ort.InferenceSession.create(location.origin+"/models/"+modelName, {
                    executionProviders: ['wasm'] //'wasm' 'webgl' 'webgpu'
                });
                console.log("Created WASM ONNX session");
            } catch(er) {console.error("WASM ONNX Create Session error:", er);}
        }
    }
}

init();



const loadImage = document.createElement('button');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext("2d", {willReadFrequently:true});
const canvas2 = document.createElement('canvas');
const ctx2 = canvas2.getContext("2d", {willReadFrequently:true});

document.body.appendChild(loadImage);
document.body.insertAdjacentHTML('beforeend','<br/>')

canvas.style.width = "50%";
document.body.appendChild(canvas);
canvas2.style.width = "50%";
document.body.appendChild(canvas2);


const testImage = async (
    image_flattened, 
    imWidth, 
    imHeight
) => {

    const start = performance.now();
    const tensor = new wgpuort.Tensor('float32', image_flattened, [1, 3, imHeight, imWidth]);
    let result = await session.run({'input':tensor});

    const duration = performance.now() - start;

    console.log("ONNX Duration: ", duration+"ms", "or", (duration/1000).toFixed(2)+"s");

    //now render result

    canvas2.width = imWidth;
    canvas2.height = imHeight;

    //render buffer
    console.log(result);

    const rgbaData = unflattenRGBImage(result.output.data, result.output.dims[3], result.output.dims[2]);

    // Create an ImageData object from the RGBA data
    const imageData = new ImageData(rgbaData, imWidth, imHeight);

    // Render the ImageData object using the canvas context
    ctx2.putImageData(imageData, 0, 0);


}



// Set the button properties for loading an image
loadImage.innerText = 'Load Image';
loadImage.addEventListener('click', () => {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.onchange = async (e) => {
        const file = e.target.files[0];
        if (file) {
            const image = new Image();
            image.onload = async () => {
                // Set canvas size to image size
                canvas.width = image.width;
                canvas.height = image.height;
                // Draw the image onto the canvas
                ctx.drawImage(image, 0, 0);

                // Extract image data
                const imageData = ctx.getImageData(0, 0, image.width, image.height);

                // Flatten and normalize the image data
                const flattenedData = convertRGBAToRGBPlanarNormalized(imageData.data, image.width, image.height);
                //todo: speed this shit up, thread it, etc

                // Pass the flattened data to the testImage function
                await testImage(flattenedData, image.width, image.height);
            };
            image.src = URL.createObjectURL(file);
        }
    };
    fileInput.click();
});






export function convertRGBAToRGBPlanar(rgbaData, outputWidth, outputHeight) {
    // Initialize the number of pixels and the output array
    const numPixels = outputWidth * outputHeight;
    const rgbData = new Float32Array(numPixels * 3);
    // Create a view of the RGBA data as 32-bit unsigned integers
    const uint32View = new Uint32Array(rgbaData.buffer);

    // Initialize indices for the R, G, and B channels in the output array
    let idxR = 0, idxG = numPixels, idxB = 2 * numPixels;

    // Loop over each pixel to convert and normalize the RGB values
    for (let i = 0; i < numPixels; i++) {
        // Extract the RGBA values using bitwise operations
        const rgba = uint32View[i];
        const r = (rgba & 0xFF) / 255.0;
        const g = ((rgba >> 8) & 0xFF) / 255.0;
        const b = ((rgba >> 16) & 0xFF) / 255.0;

        // Apply the normalization (value - mean) / std for each channel
        rgbData[idxR++] = r;
        rgbData[idxG++] = g;
        rgbData[idxB++] = b;
    }

    // Return the normalized RGB planar data
    return rgbData;
}

export function convertRGBAToRGBPlanarNormalized(rgbaData, outputWidth, outputHeight) {
        // Initialize the number of pixels and the output array
        const numPixels = outputWidth * outputHeight;
        const rgbData = new Float32Array(numPixels * 3);
    
        // Define the means and standard deviations for each channel
        let mean0 = 0.485, std0 = 0.229;
        let mean1 = 0.456, std1 = 0.224;
        let mean2 = 0.406, std2 = 0.225;
        
        // Create a view of the RGBA data as 32-bit unsigned integers
        const uint32View = new Uint32Array(rgbaData.buffer);
    
        // Initialize indices for the R, G, and B channels in the output array
        let idxR = 0, idxG = numPixels, idxB = 2 * numPixels;
    
        // Loop over each pixel to convert and normalize the RGB values
        for (let i = 0; i < numPixels; i++) {
            // Extract the RGBA values using bitwise operations
            const rgba = uint32View[i];
            const r = (rgba & 0xFF) / 255.0;
            const g = ((rgba >> 8) & 0xFF) / 255.0;
            const b = ((rgba >> 16) & 0xFF) / 255.0;
    
            // Apply the normalization (value - mean) / std for each channel
            rgbData[idxR++] = (r - mean0) / std0;
            rgbData[idxG++] = (g - mean1) / std1;
            rgbData[idxB++] = (b - mean2) / std2;
        }
    
        // Return the normalized RGB planar data
        return rgbData;
}


export function unflattenRGBImage(rgbData, outputWidth, outputHeight) {
    // Initialize the number of pixels and the output array for RGBA data
    const numPixels = outputWidth * outputHeight;
    const rgbaData = new Uint8ClampedArray(numPixels * 4);

    // // Define the means and standard deviations for each channel (must match those used for normalization)
    // let mean0 = 0.485, std0 = 0.229;
    // let mean1 = 0.456, std1 = 0.224;
    // let mean2 = 0.406, std2 = 0.225;

    // Initialize indices for the R, G, and B channels in the input array
    let idxR = 0, idxG = numPixels, idxB = 2 * numPixels;

    // Loop over each pixel to convert from normalized RGB values back to RGBA
    for (let i = 0; i < numPixels; i++) {
        // Denormalize the RGB values
        const r = (rgbData[idxR++])// * std0 + mean0) * 255;
        const g = (rgbData[idxG++])// * std1 + mean1) * 255;
        const b = (rgbData[idxB++])// * std2 + mean2) * 255;

        // Set the RGBA values, assuming full opacity for alpha
        rgbaData[i * 4] = r;
        rgbaData[i * 4 + 1] = g;
        rgbaData[i * 4 + 2] = b;
        rgbaData[i * 4 + 3] = 255; // Alpha channel
    }

    return rgbaData;
}