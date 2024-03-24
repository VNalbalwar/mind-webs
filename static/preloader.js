
let loader = document.getElementById("preloader");

window.addEventListener("load", function () {
    loader.style.transition = "1s ease-out";
    setTimeout(function () {
        loader.style.opacity = "0"; // Set opacity to 0 for fade-out effect
        // After the fade-out is complete, hide the loader
        setTimeout(function () {
            loader.style.display = "none";
            document.body.style.overflow = "visible";
        }, 1000); // Adjust the time to match the transition duration
    }, 3000);
});

