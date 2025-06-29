const fileInput = document.getElementById("imageUpload");

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
    faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
    faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
])
    .then(startApplication)
    .catch(console.error);

async function startApplication() {
    console.log("Neural models loaded successfully.");

    const wrapper = document.createElement("div");
    wrapper.style.position = "relative";
    document.body.append(wrapper);

    const labeledDescriptors = await loadKnownFaces();
    const recognizer = new faceapi.FaceMatcher(labeledDescriptors, 0.6);

    let displayedImage = null;
    let overlay = null;

    fileInput.addEventListener("change", async () => {
        if (displayedImage) displayedImage.remove();
        if (overlay) overlay.remove();

        const selectedFile = fileInput.files[0];
        if (!selectedFile) return;

        const originalImage = await faceapi.bufferToImage(selectedFile);
       
        const fixedWidth = 1000;
        const fixedHeight = 700; 

        const resizedCanvas = document.createElement("canvas");
        resizedCanvas.width = fixedWidth;
        resizedCanvas.height = fixedHeight;
        const ctx = resizedCanvas.getContext("2d");
        ctx.drawImage(originalImage, 0, 0, fixedWidth, fixedHeight);

        const resizedImage = new Image();
        resizedImage.src = resizedCanvas.toDataURL();
        await new Promise((res) => (resizedImage.onload = res));

        displayedImage = resizedImage;
        wrapper.append(displayedImage);

        overlay = faceapi.createCanvasFromMedia(displayedImage);
        wrapper.append(overlay);

        const displaySize = {
            width: displayedImage.width,
            height: displayedImage.height,
        };
        faceapi.matchDimensions(overlay, displaySize);

        const results = await faceapi
            .detectAllFaces(resizedCanvas)
            .withFaceLandmarks()
            .withFaceDescriptors();

        if (results.length === 0) {
            console.warn("Face not found.");
            return;
        }

        const resizedResults = faceapi.resizeResults(results, displaySize);

        resizedResults.forEach((result) => {
            const match = recognizer.findBestMatch(result.descriptor);
            const box = result.detection.box;
            const label = match.toString();
            const isUnknown = label.includes("unknown");

            const tag = new faceapi.draw.DrawBox(box, {
                label: label,
                boxColor: isUnknown ? "red" : "green",
                lineWidth: 2,
            });
            tag.draw(overlay);
        });
    });
}

async function loadKnownFaces() {
    const people = ["Temirlan", "Elon"];

    return Promise.all(
        people.map(async (name) => {
            const faceDescriptors = [];

            for (let imgIndex = 1; imgIndex <= 2; imgIndex++) {
                try {
                    const imgPath = `https://raw.githubusercontent.com/GalymTem/Final_project/main/labeled_images/${name}/${imgIndex}.jpg`;
                    const faceImg = await faceapi.fetchImage(imgPath);
                    const analysis = await faceapi
                        .detectSingleFace(faceImg)
                        .withFaceLandmarks()
                        .withFaceDescriptor();

                    if (!analysis) {
                        console.warn(`No face found: ${name}/${imgIndex}.jpg`);
                        continue;
                    }

                    faceDescriptors.push(analysis.descriptor);
                } catch (err) {
                    console.error(`Error processing ${name}/${imgIndex}.jpg`, err);
                }
            }

            return new faceapi.LabeledFaceDescriptors(name, faceDescriptors);
        })
    );
}
