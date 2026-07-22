document.querySelectorAll("[data-sidebar-toggle]").forEach((button) => {
  button.addEventListener("click", () => document.body.classList.toggle("sidebar-open"));
});
