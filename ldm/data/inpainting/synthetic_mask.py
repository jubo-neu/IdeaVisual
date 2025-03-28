from PIL import Image, ImageDraw
import numpy as np

settings = {
    "256narrow": {
        "p_irr": 1,
        "min_n_irr": 4,
        "max_n_irr": 50,
        "max_l_irr": 40,
        "max_w_irr": 10,
        "min_n_box": None,
        "max_n_box": None,
        "min_s_box": None,
        "max_s_box": None,
        "marg": None,
    },
    "256train": {
        "p_irr": 0.5,
        "min_n_irr": 1,
        "max_n_irr": 5,
        "max_l_irr": 200,
        "max_w_irr": 100,
        "min_n_box": 1,
        "max_n_box": 4,
        "min_s_box": 30,
        "max_s_box": 150,
        "marg": 10,
    },
    "512train": {
            "p_irr": 0.5,
            "min_n_irr": 1,
            "max_n_irr": 5,
            "max_l_irr": 450,
            "max_w_irr": 250,
            "min_n_box": 1,
            "max_n_box": 4,
            "min_s_box": 30,
            "max_s_box": 300,
            "marg": 10,
        },
    "512train-large": {
            "p_irr": 0.5,
            "min_n_irr": 1,
            "max_n_irr": 5,
            "max_l_irr": 450,
            "max_w_irr": 400,
            "min_n_box": 1,
            "max_n_box": 4,
            "min_s_box": 75,
            "max_s_box": 450,
            "marg": 10,
        },
}


def gen_segment_mask(mask, start, end, brush_width):
    mask = mask > 0
    mask = (255 * mask).astype(np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    draw.line([start, end], fill=255, width=brush_width, joint="curve")
    mask = np.array(mask) / 255
    return mask


def gen_box_mask(mask, masked):
    x_0, y_0, w, h = masked
    mask[y_0:y_0 + h, x_0:x_0 + w] = 1
    return mask


def gen_round_mask(mask, masked, radius):
    x_0, y_0, w, h = masked
    xy = [(x_0, y_0), (x_0 + w, y_0 + w)]

    mask = mask > 0
    mask = (255 * mask).astype(np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(xy, radius=radius, fill=255)
    mask = np.array(mask) / 255
    return mask


def gen_large_mask(prng, img_h, img_w,
                   marg, p_irr, min_n_irr, max_n_irr, max_l_irr, max_w_irr,
                   min_n_box, max_n_box, min_s_box, max_s_box):

    mask = np.zeros((img_h, img_w))
    uniform = prng.randint

    if np.random.uniform(0, 1) < p_irr:
        n = uniform(min_n_irr, max_n_irr)

        for _ in range(n):
            y = uniform(0, img_h)
            x = uniform(0, img_w)

            a = uniform(0, 360)
            l = uniform(10, max_l_irr)
            w = uniform(5, max_w_irr)

            x_ = x + l * np.sin(a)
            y_ = y + l * np.cos(a)

            mask = gen_segment_mask(mask, start=(x, y), end=(x_, y_), brush_width=w)
            x, y = x_, y_
    else:
        n = uniform(min_n_box, max_n_box)

        for _ in range(n):
            h = uniform(min_s_box, max_s_box)
            w = uniform(min_s_box, max_s_box)

            x_0 = uniform(marg, img_w - marg - w)
            y_0 = uniform(marg, img_h - marg - h)

            if np.random.uniform(0, 1) < 0.5:
                mask = gen_box_mask(mask, masked=(x_0, y_0, w, h))
            else:
                r = uniform(0, 60)
                mask = gen_round_mask(mask, masked=(x_0, y_0, w, h), radius=r)
    return mask


make_lama_mask = lambda prng, h, w: gen_large_mask(prng, h, w, **settings["256train"])
make_narrow_lama_mask = lambda prng, h, w: gen_large_mask(prng, h, w, **settings["256narrow"])
make_512_lama_mask = lambda prng, h, w: gen_large_mask(prng, h, w, **settings["512train"])
make_512_lama_mask_large = lambda prng, h, w: gen_large_mask(prng, h, w, **settings["512train-large"])


MASK_MODES = {
    "256train": make_lama_mask,
    "256narrow": make_narrow_lama_mask,
    "512train": make_512_lama_mask,
    "512train-large": make_512_lama_mask_large
}

if __name__ == "__main__":
    import sys

    out = sys.argv[1]

    prng = np.random.RandomState(1)
    kwargs = settings["256train"]
    mask = gen_large_mask(prng, 256, 256, **kwargs)
    mask = (255 * mask).astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.save(out)
