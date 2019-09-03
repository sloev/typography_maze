import draw
import sys
from afinn import Afinn
from tqdm import tqdm

afinn = Afinn(language="en")

with open(sys.argv[1]) as f:
    width = 1000000
    height = 1000000
    x = width/2
    y = height/2
    with draw.create_canvas_render(width, height, x, y, dry=True) as (render, points):
        for word in tqdm(f.read().split(" ")):
            polarity = max(min(int(afinn.score(word)), 1), -1)
            render(f" {word} ", polarity)

        x_points = sorted(points, key=lambda x: x[0])
        y_points = sorted(points, key=lambda x: x[1])

        x_range = x_points[0][0], x_points[-1][0]
        y_range = y_points[0][1], y_points[-1][1]

    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]

    x -= x_range[0]
    y -= y_range[0]

    print(f"running for real with x:{x}, y:{y}, width:{width}, height:{height}")
    f.seek(0)
    with draw.create_canvas_render(width, height, x, y) as render:
        for word in tqdm(f.read().split(" ")):
            polarity = max(min(int(afinn.score(word)), 1), -1)
            render(f" {word} ", polarity)
