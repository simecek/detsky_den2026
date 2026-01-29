"""
Kresba na obrázek – aplikace

Gradio aplikace, která promění dětské kresby na krásné obrázky
pomocí AI generování obrázků (OpenAI nebo Gemini).
"""

import subprocess
import tempfile

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from providers import get_provider

# Předdefinované umělecké styly (český popisek, anglický klíč pro model)
STYLES = [
    ("kreslený / animovaný", "cartoon/animated"),
    ("akvarelová malba", "watercolor painting"),
    ("olejomalba", "oil painting"),
    ("digitální ilustrace", "digital art"),
    ("3D obrázek", "3D rendered"),
    ("pixel art", "pixel art"),
    ("anime / manga", "anime/manga"),
    ("realistická fotografie", "realistic photograph"),
    ("tužková kresba (vylepšená)", "pencil sketch (refined)"),
    ("pohádková ilustrace", "storybook illustration"),
    ("pop art", "pop art"),
    ("kubismus", "cubism"),
    ("styl Krteček (pohádka Zdeněk Miler)", "fairy tale Little Mole (Zdenek Miler)"),
    ("Josef Lada", "Josef Lada like"),
    ("Alfons Mucha", "Alfond Mucha like"),
    
]


def transform_sketch(
    sketch: Image.Image,
    style: str,
    custom_prompt: str,
    progress=gr.Progress(),
) -> Image.Image:
    """
    Přeměň kresbu na obrázek ve zvoleném stylu.

    Args:
        sketch: Vstupní kresba / obrázek
        style: Umělecký styl
        custom_prompt: Doplňující instrukce pro AI

    Returns:
        Vygenerovaný obrázek
    """
    if sketch is None:
        raise gr.Error("Nejprve nahraj kresbu!")

    if not style:
        raise gr.Error("Vyber styl!")

    progress(0.1, desc="Inicializuji AI...")

    try:
        provider = get_provider("gemini")
    except Exception as e:
        raise gr.Error(f"Nepodařilo se spustit poskytovatele: {e}")

    progress(0.3, desc=f"Vytvářím obrázek pomocí {provider.name}...")

    try:
        result = provider.generate_from_sketch(
            sketch=sketch,
            style=style,
            prompt=custom_prompt if custom_prompt.strip() else None,
        )
        progress(1.0, desc="Hotovo!")
        return result
    except Exception as e:
        raise gr.Error(f"Generování obrázku selhalo: {e}")


def create_print_layout(
    original: Image.Image, generated: Image.Image
) -> Image.Image:
    """
    Vytvoř tiskový layout s oběma obrázky pro A4.

    Args:
        original: Originální kresba
        generated: Vygenerovaný obrázek

    Returns:
        Kompozitní obrázek pro tisk
    """
    # A4 při 150 DPI (dostatečné pro tisk, menší soubor)
    A4_WIDTH = 1240  # 210mm
    A4_HEIGHT = 1754  # 297mm
    MARGIN = 60
    LABEL_HEIGHT = 50
    SPACING = 40

    # Vytvoř bílé A4 plátno
    canvas = Image.new("RGB", (A4_WIDTH, A4_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)

    # Pokus se načíst systémový font, jinak výchozí
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Vypočítej dostupný prostor pro každý obrázek
    available_width = A4_WIDTH - 2 * MARGIN
    available_height = (A4_HEIGHT - 2 * MARGIN - 2 * LABEL_HEIGHT - SPACING) // 2

    def resize_to_fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
        """Změň velikost obrázku tak, aby se vešel do zadaného prostoru."""
        ratio = min(max_w / img.width, max_h / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        return img.resize(new_size, Image.Resampling.LANCZOS)

    # Zpracuj originální obrázek
    orig_resized = resize_to_fit(original, available_width, available_height)
    orig_x = MARGIN + (available_width - orig_resized.width) // 2
    orig_y = MARGIN + LABEL_HEIGHT

    # Zpracuj vygenerovaný obrázek
    gen_resized = resize_to_fit(generated, available_width, available_height)
    gen_x = MARGIN + (available_width - gen_resized.width) // 2
    gen_y = MARGIN + LABEL_HEIGHT + available_height + SPACING + LABEL_HEIGHT

    # Nakresli popisky
    draw.text((MARGIN, MARGIN), "Originál:", fill="black", font=font)
    draw.text(
        (MARGIN, MARGIN + LABEL_HEIGHT + available_height + SPACING),
        "Vygenerováno:",
        fill="black",
        font=font,
    )

    # Vlož obrázky
    canvas.paste(orig_resized, (orig_x, orig_y))
    canvas.paste(gen_resized, (gen_x, gen_y))

    return canvas


def print_images(original: Image.Image, generated: Image.Image) -> str:
    """
    Vytiskni oba obrázky na výchozí tiskárně.

    Args:
        original: Originální kresba
        generated: Vygenerovaný obrázek

    Returns:
        Zpráva o stavu tisku
    """
    if original is None:
        raise gr.Error("Chybí originální kresba!")
    if generated is None:
        raise gr.Error("Nejprve vygeneruj obrázek!")

    try:
        # Vytvoř kompozitní obrázek
        print_layout = create_print_layout(original, generated)

        # Ulož do dočasného souboru
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            print_layout.save(f.name)
            # Tisk pomocí výchozí tiskárny
            result = subprocess.run(
                ["lpr", f.name],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise gr.Error(f"Tisk selhal: {result.stderr}")
            return "Obrázek odeslán na tiskárnu!"
    except FileNotFoundError:
        raise gr.Error("Příkaz 'lpr' nenalezen. Je tiskárna nastavena?")
    except Exception as e:
        raise gr.Error(f"Chyba při tisku: {e}")


def create_app() -> gr.Blocks:
    """Vytvoř a nastav Gradio aplikaci."""

    with gr.Blocks(title="Kresba na obrázek") as app:
        gr.Markdown(
            """
            # Proměň kresbu na obrázek

            Nahraj kresbu (třeba dětský obrázek) a sleduj, jak se promění
            v krásný obrázek ve vybraném stylu!

            **Jak to funguje:**
            1. Nahraj svou kresbu (fotka obrázku funguje skvěle!)
            2. Vyber umělecký styl
            3. (Volitelně) napiš vlastní instrukce
            4. Klikni na „Proměnit!“
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Vstupní část
                sketch_input = gr.Image(
                    label="Nahraj svou kresbu",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=400,
                )

                style_dropdown = gr.Dropdown(
                    choices=STYLES,
                    value=STYLES[0][1],
                    label="Vyber styl",
                    info="Vyber, jak se má kresba proměnit",
                )

                custom_prompt = gr.Textbox(
                    label="Vlastní instrukce (volitelné)",
                    placeholder="např. 'Použij veselé barvy' nebo 'Přidej kouzelný les na pozadí'",
                    lines=2,
                )

                transform_btn = gr.Button(
                    "Proměnit!",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                # Výstupní část
                output_image = gr.Image(
                    label="Vygenerovaný obrázek",
                    type="pil",
                    height=400,
                )

                regenerate_btn = gr.Button(
                    "🔄 Přegenerovat",
                    variant="secondary",
                    size="lg",
                )

                print_btn = gr.Button(
                    "🖨️ Vytisknout",
                    variant="secondary",
                    size="lg",
                )

                print_status = gr.Textbox(
                    label="Stav tisku",
                    interactive=False,
                    visible=False,
                )

        # Sekce ukázek
        gr.Markdown("### Ukázky stylů")
        gr.Markdown(
            "Vyzkoušej různé styly a sleduj, jak se kresba mění! "
            "Pro dětské obrázky se skvěle hodí pohádkový nebo kreslený styl."
        )

        # Připojení tlačítek
        transform_btn.click(
            fn=transform_sketch,
            inputs=[sketch_input, style_dropdown, custom_prompt],
            outputs=output_image,
        )

        print_btn.click(
            fn=print_images,
            inputs=[sketch_input, output_image],
            outputs=[print_status],
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[print_status],
        )

        regenerate_btn.click(
            fn=transform_sketch,
            inputs=[sketch_input, style_dropdown, custom_prompt],
            outputs=output_image,
        )

    return app


def main():
    """Spusť Gradio aplikaci."""
    app = create_app()
    app.launch(
        share=True,
        show_error=True,
        max_threads=5,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
        ),
    )


if __name__ == "__main__":
    main()
