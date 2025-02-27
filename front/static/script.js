document.addEventListener("DOMContentLoaded", () => {
    const imageSection = document.getElementById("image-section");
    const preprocessSection = document.getElementById("preprocess-section");
    const textSection = document.getElementById("text-section");
    const fullTextSection = document.getElementById("fulltext-section");
    const checkSection = document.getElementById("check-section");
    const linesImagesDiv = document.getElementById("lines-images");
    const extractionSection = document.getElementById("extraction-section");

    // Image quality buttons
    document.getElementById("quality-yes").addEventListener("click", () => {
        fetch("/check-image-quality", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ decision: "yes" })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === "preprocessed") {
                document.getElementById("preprocessed-image").src = data.image;
                linesImagesDiv.innerHTML = "";  // Clear any previous images
                const list = document.createElement("ol");

                data.lines.forEach(lineImagePath => {
                    let element = document.createElement("li");
                    element.style.alignItems = "left"

                    const img = document.createElement("img");
                    img.src = lineImagePath;
                    img.alt = "Detected line image";
                    element.appendChild(img);
                    list.appendChild(element)
                });
                linesImagesDiv.appendChild(list)
                imageSection.style.display = "none";
                preprocessSection.style.display = "block";
            }
        });
    });

    // Event listener for the Recrop button
    document.getElementById("recrop-button").addEventListener("click", () => {
        const projectionValue = document.getElementById("projection-value").value;

        // Validate projection value
        if (!projectionValue || isNaN(projectionValue)) {
            alert("Please enter a valid projection value.");
            return;
        }

        fetch("/crop_with_projection", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ projectionValue: Number(projectionValue) }) // Send projection value to the server
        })
            .then(response => response.json())
            .then(data => {
                // Replace the linesImagesDiv content with new cropped lines
                linesImagesDiv.innerHTML = ""; // Clear previous images
                const list = document.createElement("ol");

                data.lines.forEach(lineImagePath => {
                    let element = document.createElement("li");
                    element.style.alignItems = "left";

                    const img = document.createElement("img");
                    img.src = lineImagePath;
                    img.alt = "Detected line image";
                    element.appendChild(img);
                    list.appendChild(element);
                });

                linesImagesDiv.appendChild(list);
            })
            .catch(error => {
                console.error("Error recropping the image:", error);
                alert("An error occurred. Please try again.");
            });
    });


    document.getElementById("quality-no").addEventListener("click", () => {
        if (window.confirm("Do you want to retake the image?")) {
            // User pressed "Yes"
            location.reload(); // Reload page or update the UI as necessary
        }
    });
    document.getElementById("reload-image").addEventListener("click", () => {
        location.reload(); // Reload page or update the UI as necessary
    });

    document.getElementById("try-again").addEventListener("click", () => {
        if (window.confirm("Do you want to retake the image?")) {
            // User pressed "Yes"
            location.reload(); // Reload page or update the UI as necessary
        }
    });

    document.getElementById("preprocess-no").addEventListener("click", () => {
        if (window.confirm("Do you want to retake the image?")) {
            // User pressed "Yes"
            location.reload(); // Reload page or update the UI as necessary
        }
    });
    document.getElementById("cut-no").addEventListener("click", () => {
        if (window.confirm("Do you want to retake the image?")) {
            // User pressed "Yes"
            location.reload(); // Reload page or update the UI as necessary
        }
    });

    const devicesForm = document.querySelector('form[action="/set-device"]');
    const devicesDropdown = document.getElementById("devices");

    devicesForm.addEventListener("submit", (event) => {
        event.preventDefault(); // Prevent the form from submitting traditionally

        const selectedDevice = devicesDropdown.value;

        fetch("/set-device", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ device: selectedDevice }),
        })
        .then((response) => response.json())
        .then((data) => {
            if (data.status === false) {
                // Show the error message to the user
                alert(`Error: ${data.reason}`);

                // Revert the dropdown selection to "CPU"
                devicesDropdown.value = "cpu"; // Assuming "audi" corresponds to "CPU"
            }
            else {
                const lineImages = Array.from(linesImagesDiv.getElementsByTagName("img")).map(img => img.src);
                const submitButton = document.querySelector('input[type="submit"]');

                // Change the button text to "Extracting..." and disable it
                submitButton.value = "Extracting...";
                submitButton.disabled = true;
                submitButton.style.background = "lightgrey"
                // Call the server to extract text from these images
                fetch("/extract-text", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ lines: lineImages, decision: "yes" }) // Send the image paths
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === "text_extracted") {
                        // Create a container for displaying images and extracted text
                        const textContainer = document.createElement("div");
                        textContainer.style.display = "flex";
                        textContainer.style.flexWrap = "wrap";
                        textContainer.style.gap = "20px";
                        textContainer.style.justifyContent = "center";

                        let full_text = "";

                        data.text.forEach((text, index) => {
                            const column = document.createElement("div");
                            column.style.display = "flex";
                            column.style.flexDirection = "column";
                            column.style.alignItems = "center";

                            // Original image
                            const originalImage = document.createElement("img");
                            originalImage.src = lineImages[index];
                            originalImage.alt = "Original line image";
                            column.appendChild(originalImage);

                            // Modifiable text box
                            const textBox = document.createElement("p");
                            textBox.innerHTML = text;  // Pre-fill with extracted text
                            textBox.style.margin = "0"; // Remove the gap
                            column.appendChild(textBox);

                            textContainer.appendChild(column);
                            full_text += text + "\n";
                        });


                        // Modifiable text box for full text
                        const fullTextBox = document.createElement("textarea");
                        fullTextBox.value = full_text; // Pre-fill with extracted text
                        fullTextSection.appendChild(fullTextBox);

                        // Append the textContainer to the lines-images div
                        textSection.appendChild(textContainer);

                        // Show the text extraction section
                        preprocessSection.style.display = "none";
                        linesImagesDiv.style.display = "none";
                        extractionSection.style.display = "block";
                    }
                });

            }
        })
        .catch((error) => {
            console.error("An error occurred:", error);
            alert("An unexpected error occurred. Please try again.");
            // Revert the dropdown selection to "CPU" in case of a failure
            devicesDropdown.value = "cpu";
        });
    });

    // Syntax and Grammar check button
    document.getElementById("syntax-check").addEventListener("click", () => {
        const fullText = document.getElementById("fulltext-section").querySelector("textarea").value;

        const submitButton = document.getElementById("syntax-check")

        // Change the button text to "Extracting..." and disable it
        submitButton.disabled = true;
        submitButton.style.background = "lightgrey"

        // First, perform the syntax check
        fetch("/check-syntax-and-grammar", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: fullText })
        })
        .then(response => response.json())
        .then(data => {
            const syntaxFixed = data.syntaxFixed;
            const grammarFixed = data.grammarFixed;
            //
            // // Then, perform the grammar check on the syntax-fixed version
            // fetch("/check-grammar", {
            //     method: "POST",
            //     headers: { "Content-Type": "application/json" },
            //     body: JSON.stringify({ text: syntaxFixed })
            // })
            // .then(response => response.json())
            // .then(data => {
            //     const grammarFixed = data.fixed;

            // Show both the syntax and grammar fixed versions
            document.getElementById("check-original").textContent = fullText;
            document.getElementById("check-syntax-fixed").textContent = syntaxFixed;
            document.getElementById("check-grammar-fixed").textContent = grammarFixed;

            // Show the combined check section
            extractionSection.style.display = "none";
            checkSection.style.display = "block";
            // });
        });
    });
    document.getElementById("back-preprocessing").addEventListener("click", () => {
        checkSection.style.display = "none"; // Hide the check section
        extractionSection.style.display = "block"; // Show the extraction section
    });
});
