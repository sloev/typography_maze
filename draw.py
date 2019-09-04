import cairocffi

cairocffi.install_as_pycairo()
import cairo
import io
import math
from contextlib import contextmanager


@contextmanager
def create_canvas_render(width, height, x, y, font_size, filename="output", dry=False):
    TURNS = {-1: -math.pi / 2, 0: 0, 1: math.pi / 2}
    if dry:
        device = io.BytesIO()
    else:
        device = open(filename + ".svg", "wb")

    surface = cairo.SVGSurface(device, width, height)

    context = cairo.Context(surface)
    context.rectangle(0, 0, width, height)
    context.set_source_rgb(1, 1, 1)
    context.fill()
    context.set_source_rgb(0, 0, 0)

    context.select_font_face("times", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    context.set_font_size(font_size)

    context.move_to(x, y)

    data = dict(
        direction=0, points=[context.user_to_device(*context.get_current_point())]
    )

    def render(text, turn=0):
        data["direction"] += turn
        if data["direction"] < 0:
            data["direction"] = 3
        if data["direction"] > 3:
            data["direction"] = 0

        context.rotate(TURNS[turn])
        _, _, width, _, _, _ = context.text_extents(text)
        context.show_text(text)

        data["points"].append(context.user_to_device(*context.get_current_point()))

    if dry:
        yield render, data["points"]
    else:
        yield render

    context.fill()
    context.stroke()

    surface.finish()
    device.close()
