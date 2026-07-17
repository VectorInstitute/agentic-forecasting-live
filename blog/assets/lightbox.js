// Dependency-free click-to-enlarge lightbox for figure images.
// No CDNs, no external requests -- inlined into the rendered HTML by render.sh.
(function () {
  "use strict";
  var overlay = document.getElementById("lightbox");
  if (!overlay) return;
  var overlayImg = overlay.querySelector("img");
  var closeBtn = overlay.querySelector(".lightbox-close");
  var lastFocused = null;

  function open(img) {
    lastFocused = document.activeElement;
    overlayImg.src = img.currentSrc || img.src;
    overlayImg.alt = img.alt || "";
    overlay.classList.add("open");
    document.body.style.overflow = "hidden";
    closeBtn.focus();
  }

  function close() {
    overlay.classList.remove("open");
    overlayImg.src = "";
    document.body.style.overflow = "";
    if (lastFocused && lastFocused.focus) lastFocused.focus();
  }

  document.querySelectorAll("figure.fig img").forEach(function (img) {
    img.setAttribute("tabindex", "0");
    img.setAttribute("role", "button");
    img.setAttribute("aria-label", "Click to enlarge image");
    img.addEventListener("click", function () {
      open(img);
    });
    img.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        open(img);
      }
    });
  });

  overlay.addEventListener("click", function (e) {
    if (e.target === overlay) close();
  });
  closeBtn.addEventListener("click", close);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && overlay.classList.contains("open")) close();
  });
})();
