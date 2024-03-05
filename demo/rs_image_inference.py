# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import RSImage, RSInferencer


def main():
    parser = ArgumentParser()
    parser.add_argument('image', help='Image file path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--output-path',
        help='Path to save result image',
        default='result.png')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='maximum number of windows inferred simultaneously')
    parser.add_argument(
        '--window-size',
        help='window xsize,ysize',
        default=(224, 224),
        type=int,
        nargs=2)
    parser.add_argument(
        '--stride',
        help='window xstride,ystride',
        default=(224, 224),
        type=int,
        nargs=2)
    parser.add_argument(
        '--color_map',
        help='color mapping of segment classes',
        default=[
    (0, 0, 0), (128, 0, 0),  (0, 255, 36), (148, 148, 148),  (255, 255, 255),
    (34, 97, 38),  (0, 69, 255), (75, 181, 73),  (222, 31, 7)
],
        type=list)
    parser.add_argument(
        '--thread', default=1, type=int, help='number of inference threads')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    inferencer = RSInferencer.from_config_path(
        args.config,
        args.checkpoint,
        batch_size=args.batch_size,
        thread=args.thread,
        device=args.device)
    image = RSImage(args.image)

    inferencer.run(image, args.window_size, args.stride, args.output_path, args.color_map)


if __name__ == '__main__':
    main()
