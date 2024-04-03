
from shiny.express import input, render, ui
from shinywidgets import render_widget


with ui.sidebar():
    ui.input_select("var", "Select variable", choices=["total_bill", "tip"])

    ui.panel_title("Hello Shiny!")
    ui.input_slider("n", "N", 0, 100, 20)
    ui.input_slider("agents", "Agents", 0, 100, 100),
    ui.input_slider("steps", "Steps", 0, 100, 100),
    ui.input_slider("x", "y", 0, 10, 10),
    ui.input_slider("y", "y", 0, 10, 10),
    ui.input_selectize(
        "var", "Select variable",
        choices=["bill_length_mm", "body_mass_g"]
    )


@render.text
def txt():
    return f"n*2 is {input.n() * 2}"

@render.plot
def hist():
    from matplotlib import pyplot as plt
    from palmerpenguins import load_penguins

    df = load_penguins()
    df[input.var()].hist(grid=False)
    plt.xlabel(input.var())
    plt.ylabel("count")

