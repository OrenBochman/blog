// slide-margin.js
// Bootstrap-adapted light-DOM slide component for Quarto fenced divs.
// Usage:
//
// :::: {.sl}
// ::: {.sl-text}
// Markdown text.
// :::
//
// ::: {.sl-fig}
// ![Caption](slide02.png)
// :::
// ::::

const css = `
.sl,
blog-slide {
  --sl-body-cols: 8;
  --sl-total-cols: 12;
  --sl-bleed: calc(var(--sl-total-cols) / var(--sl-body-cols) * 100%);

  display: block;
  position: relative;
  width: var(--sl-bleed);
  max-width: none;
  overflow: visible;
  margin-block: 1.5rem;
}

.sl-row {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  width: 100%;
  max-width: none;
  margin-right: 0;
  margin-left: 0;
}

.sl-row > .sl-text {
  flex: 0 0 66.6666666667%;
  max-width: 66.6666666667%;
  min-width: 0;
  padding-right: 1rem;
}

.sl-row > .sl-fig {
  flex: 0 0 33.3333333333%;
  max-width: 33.3333333333%;
  min-width: 0;
  padding-left: 1rem;
}

.sl-fig p,
.sl-fig figure,
.sl-fig .quarto-figure,
.sl-fig .quarto-figure-center {
  margin: 0 !important;
  padding: 0 !important;
}

.sl-fig figcaption {
  font-size: 0.8rem;
  line-height: 1.25;
  margin-top: 0.35rem;
}

.sl-fig img {
  display: block !important;
  width: 100% !important;
  max-width: 100% !important;
  height: auto !important;
  vertical-align: top !important;
  border-radius: 0.4rem;
}

@media (max-width: 767.98px) {
  .sl,
  blog-slide {
    width: 100%;
  }

  .sl-row > .sl-text,
  .sl-row > .sl-fig {
    flex: 0 0 100%;
    max-width: 100%;
    padding-inline: 0;
  }

  .sl-row > .sl-fig {
    margin-top: 1rem;
  }
}
`;

function installStyles() {
  if (document.getElementById("slide-margin-styles")) return;

  const style = document.createElement("style");
  style.id = "slide-margin-styles";
  style.textContent = css;
  document.head.appendChild(style);
}

function normalizeSlideLightbox(fig, gallery = "slides") {
  fig.querySelectorAll("a.lightbox").forEach((a) => {
    a.dataset.gallery = gallery;
    a.setAttribute("data-gallery", gallery);

    const img = a.querySelector("img");

    if (img) {
      if (!a.getAttribute("href")) {
        a.setAttribute("href", img.currentSrc || img.src);
      }

      if (!a.getAttribute("title") && img.alt) {
        a.setAttribute("title", img.alt);
      }

      if (!a.getAttribute("data-original-href")) {
        a.setAttribute("data-original-href", a.getAttribute("href"));
      }
    }
  });

  fig.querySelectorAll("img").forEach((img) => {
    const existing = img.closest("a");

    if (existing) {
      existing.classList.add("lightbox");
      existing.dataset.gallery = gallery;
      existing.setAttribute("data-gallery", gallery);

      if (!existing.getAttribute("href")) {
        existing.setAttribute("href", img.currentSrc || img.src);
      }

      if (!existing.getAttribute("title") && img.alt) {
        existing.setAttribute("title", img.alt);
      }

      if (!existing.getAttribute("data-original-href")) {
        existing.setAttribute("data-original-href", existing.getAttribute("href"));
      }

      return;
    }

    const a = document.createElement("a");
    a.className = "lightbox";
    a.href = img.currentSrc || img.src;
    a.dataset.gallery = gallery;
    a.setAttribute("data-gallery", gallery);
    a.setAttribute("data-original-href", a.href);

    if (img.alt) {
      a.title = img.alt;
    }

    img.replaceWith(a);
    a.appendChild(img);
  });
}

function upgradeOne(el) {
  if (el.dataset.slReady === "true") {
    const fig = el.querySelector(".sl-fig");
    if (fig) normalizeSlideLightbox(fig, "slides");
    return;
  }

  const text = el.querySelector(":scope > .sl-text, .sl-text");
  const fig = el.querySelector(":scope > .sl-fig, .sl-fig");

  if (!text || !fig) return;

  normalizeSlideLightbox(fig, "slides");

  const row = document.createElement("div");
  row.className = "sl-row row";

  text.classList.add("col-8");
  fig.classList.add("col-4");

  fig.querySelectorAll(".col-margin").forEach((x) => {
    x.classList.remove("col-margin");
  });

  row.append(text, fig);
  el.replaceChildren(row);

  el.dataset.slReady = "true";
}

class BlogSlide extends HTMLElement {
  connectedCallback() {
    upgradeOne(this);
  }
}

if (!customElements.get("blog-slide")) {
  customElements.define("blog-slide", BlogSlide);
}

function upgradeSlides(root = document) {
  installStyles();

  root.querySelectorAll("div.sl").forEach((el) => {
    if (el.dataset.slReady === "true") return;

    const component = document.createElement("blog-slide");

    for (const attr of el.attributes) {
      if (attr.name !== "class") {
        component.setAttribute(attr.name, attr.value);
      }
    }

    while (el.firstChild) {
      component.appendChild(el.firstChild);
    }

    el.replaceWith(component);
    upgradeOne(component);
  });

  root.querySelectorAll("blog-slide").forEach(upgradeOne);
}

installStyles();

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => upgradeSlides());
} else {
  upgradeSlides();
}

requestAnimationFrame(() => upgradeSlides());
setTimeout(() => upgradeSlides(), 250);

const observer = new MutationObserver(() => upgradeSlides());
observer.observe(document.documentElement, {
  childList: true,
  subtree: true
});