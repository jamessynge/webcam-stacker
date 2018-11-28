#!/usr/bin/env python3.6
"""
Initial experiment in stacking a set of captured snapshots.
"""

import argparse
import glob
import io
import logging
import os
import re
import sys

import numpy as np
from PIL import Image
import skimage


class NotEqualException(Exception):
    pass


def check_equal(a, b):
    if a != b:
        raise NotEqualException(f'{a} != {b}')


class WrongTypeException(Exception):
    pass


def check_types(value, classinfo):
    if not isinstance(value, classinfo):
        raise WrongTypeException(
            f'Value {value} (type {type(value)} is not an instance of {classinfo}')


def jpeg_bytes_to_ndarray(jpeg_bytes):
    #print('len(jpeg_bytes):', len(jpeg_bytes))
    image = np.array(Image.open(io.BytesIO(jpeg_bytes)))
    # print('raw histogram:')
    # print(np.histogram(image, bins=256))
    # sys.exit(1)

    # print('Image in num:\n', image)
    return image


def fit_ndarray_to_range(array, in_min=np.amin, out_min=0.0, in_max=np.amax, out_max=None, in_place=False):
    if out_max is None:
        raise ValueError('out_max must be a number')

    if callable(in_min):
        in_min = in_min(array)
    if callable(in_max):
        in_max = in_max(array)

    # array is in range [in_min, in_max]
    scale = (out_max - out_min) / (in_max - in_min)

    # Shift so that minimum value in the array is zero.
    if in_min != 0:
        if in_place:
            np.subtract(array, in_min, out=array)
        else:
            array = np.subtract(array, in_min)
            # Operations can be in place after this.
            in_place = True

    # array is in range [0, in_max - in_min]

    if scale != 0:
        if in_place:
            np.multiply(array, scale, out=array)
        else:
            array = np.multiply(array, scale)
            # Operations can be in place after this.
            in_place = True

    # array is in range [0, out_max - out_min]

    if out_min != 0:
        if in_place:
            np.add(array, out_min, out=array)
        else:
            array = np.add(array, out_min)

    # array is in range [out_min, out_max]
    return array



    # lower_to_func=np.amin, scale_func=lambda a: 255.999 / np.amax(a), inplace=False):
    # if lower_to_func:
    #     lower_to = lower_to_func(array)
    #     if inplace:
    #         np.subtract(array, lower_to, out=array)
    #     else:
    #         array = np.subtract(array, lower_to)
    #         inplace = True

    # if scale_func:
    #     scale = scale_func(array)
    #     if inplace:
    #         np.multiply(array, scale, out=array)
    #     else:
    #         array = np.multiply(array, scale)

    # return array


def ndarray_to_pil_image(array):
    array = fit_ndarray_to_range(array, out_min=0.0, out_max=255.999)
    byte_array = array.astype('uint8')
    byte_image = Image.fromarray(byte_array)
    return byte_image


class HsvImageStacker(object):
    """docstring for HsvImageStacker"""

    def __init__(self):
        super(HsvImageStacker, self).__init__()
        self.stacked_hsv_image = None
        self.hsv_images = []

    def stack_depth(self):
        return len(self.hsv_images)

    def add_rgb_ndarray(self, rgb_image):
        check_types(rgb_image, np.ndarray)
        # Convert to HSV color space.
        hsv_image = skimage.color.rgb2hsv(rgb_image)

        # # DEBUGGING:
        # print(f'rgb_image shape={rgb_image.shape} amin={np.amin(rgb_image)}, amax={np.amax(rgb_image)}')
        # print('hsv_image shape:', hsv_image.shape)
        # print('hsv_image amin:', np.amin(hsv_image))
        # print('hsv_image amax:', np.amax(hsv_image))
        # h = hsv_image[:,:,0]
        # s = hsv_image[:,:,1]
        # v = hsv_image[:,:,2]
        # print(f'hsv_image h shape={h.shape} amin={np.amin(h):6.4f}, amax={np.amax(h):6.4f}')
        # print(f'hsv_image s shape={s.shape} amin={np.amin(s):6.4f}, amax={np.amax(s):6.4f}')
        # print(f'hsv_image v shape={v.shape} amin={np.amin(v):6.4f}, amax={np.amax(v):6.4f}')

        self.hsv_images.append(hsv_image)
        if self.stacked_hsv_image is None:
            self.stacked_hsv_image = hsv_image.astype('float64')  # Maybe copy it.
        else:
            check_equal(hsv_image.shape, self.stacked_hsv_image.shape)
            np.add(self.stacked_hsv_image, hsv_image, out=self.stacked_hsv_image)

        h = self.stacked_hsv_image[:,:,0]
        s = self.stacked_hsv_image[:,:,1]
        v = self.stacked_hsv_image[:,:,2]
        print(f'stacked_hsv_image h shape={h.shape} amin={np.amin(h):6.4f}, amax={np.amax(h):6.4f}')
        print(f'stacked_hsv_image s shape={s.shape} amin={np.amin(s):6.4f}, amax={np.amax(s):6.4f}')
        print(f'stacked_hsv_image v shape={v.shape} amin={np.amin(v):6.4f}, amax={np.amax(v):6.4f}')

    def add_pil_image(self, pil_image):
        check_types(pil_image, Image.Image)
        rgb_image = np.array(pil_image)
        self.add_rgb_ndarray(rgb_image)

    def add_image_file(self, file_path):
        check_types(file_path, str)
        with open(file_path, 'rb') as f:
            pil_image = Image.open(io.BytesIO(f.read()))
        self.add_pil_image(pil_image)

    def remove_oldest(self):
        oldest = self.hsv_images.pop(0)
        np.subtract(self.stacked_hsv_image, oldest, out=self.stacked_hsv_image)
        return oldest

    # def scale(self, scale):
    #     np.multiply(self.stacked_hsv_image, scale, out=self.stacked_hsv_image)

    def get_unit_hsv_image(self):
        return self.stacked_hsv_image / self.stack_depth()

    def get_byte_image(self):
        """Return an image with channel values in the range [0, 255]."""
        # print('stacked_image histogram')
        # print(np.histogram(self.stacked_hsv_image, bins=256))

        # Subtract the minimum value so that the values start at zero (black).

        zero_image = self.stacked_hsv_image - np.amin(self.stacked_hsv_image)

        # print('zero_image histogram')
        # print(np.histogram(zero_image, bins=256))

        # Now divide by the maximum value / 255.5, bringing the max value to 255.5.
        zero_image = zero_image.astype('double')
        byte_image = zero_image / (np.amax(zero_image) / 255.0)

        # print('byte_image histogram')
        # print(np.histogram(byte_image, bins=256))

        return byte_image

    def equalize_hist(self, nbins=256):
        print('stacked_image shape:', self.stacked_hsv_image.shape)
        print('stacked_image amin:', np.amin(self.stacked_hsv_image))
        print('stacked_image amax:', np.amax(self.stacked_hsv_image))

        image = exposure.equalize_hist(
            self.stacked_hsv_image, nbins=nbins)
        print('equalize_hist image shape:', image.shape)
        print('equalize_hist image amin:', np.amin(image))
        print('equalize_hist image amax:', np.amax(image))

        byte_image = image / (np.amax(image) / 255.0)
        return byte_image

    def equalize_adapthist(self, kernel_size=None, clip_limit=0.01, nbins=256):
        print('stacked_image shape:', self.stacked_hsv_image.shape)
        print('stacked_image amin:', np.amin(self.stacked_hsv_image))
        print('stacked_image amax:', np.amax(self.stacked_hsv_image))

        image = exposure.equalize_adapthist(
            self.stacked_hsv_image, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins)
        print('adapthist image shape:', image.shape)
        print('adapthist image amin:', np.amin(image))
        print('adapthist image amax:', np.amax(image))

        byte_image = image / (np.amax(image) / 255.0)
        return byte_image


def process_images(image_paths, output_pattern, min_depth, depth):
    image_paths.sort()
    stacker = HsvImageStacker()
    dirs = {}
    for n, fn in enumerate(image_paths):
        print('Loading', fn)
        stacker.add_image_file(fn)

        while stacker.stack_depth() > depth:
            stacker.remove_oldest()

        if stacker.stack_depth() < min_depth:
            continue

        # print('stacker.stacked_image histogram')
        # print(np.histogram(stacker.stacked_image, bins=256))

        unit_hsv_image = stacker.get_unit_hsv_image()
        # h = unit_hsv_image[:,:,0]
        # s = unit_hsv_image[:,:,1]
        v = unit_hsv_image[:,:,2]

        # print('unit_hsv_image v histogram RAW')
        # print(np.histogram(v, bins=256))

        # fit_ndarray_to_range(v, out_min=0.0, out_max=1.0, in_place=True)

        print('unit_hsv_image v histogram BEFORE equalize_*')
        histogram_before = np.histogram(v, bins=256)
        print(np.histogram(v, bins=256))

        # unit_hsv_image[:,:,2] = skimage.exposure.equalize_adapthist(
        #     unit_hsv_image[:,:,2], nbins=256, clip_limit=0.1)
        unit_hsv_image[:,:,2] = skimage.exposure.equalize_hist(
            unit_hsv_image[:,:,2], nbins=256)

        print('unit_hsv_image v histogram AFTER')
        print(np.histogram(v, bins=256))


        print('unit_hsv_image v histogram AFTER - BEFORE')
        print(np.histogram(v, bins=256)[0] - histogram_before[0])



        unit_rgb_image = skimage.color.hsv2rgb(unit_hsv_image)
        byte_image = ndarray_to_pil_image(unit_rgb_image)
        byte_image.show()

        return

        # print('unit_image g histogram')
        # print(np.histogram(g, bins=256))

        # print('unit_image b histogram')
        # print(np.histogram(b, bins=256))

        # hr = np.histogram(r, bins=256)
        # hg = np.histogram(g, bins=256)
        # hb = np.histogram(b, bins=256)

        er = exposure.equalize_hist(r, nbins=256)
        eg = exposure.equalize_hist(g, nbins=256)
        eb = exposure.equalize_hist(b, nbins=256)


        # print('equalized r histogram')
        # print(np.histogram(er, bins=256))

        # print('equalized g histogram')
        # print(np.histogram(eg, bins=256))

        # print('equalized b histogram')
        # print(np.histogram(eb, bins=256))

        e_all = (er + eg + eb)

        # print('e_all histogram')
        # print(np.histogram(e_all, bins=3*256))
        # return

        # unit_e_all = e_all / np.amax(e_all)

        # print('unit_e_all.shape:', unit_e_all.shape)
        # print('unit_e_all histogram')
        # print(np.histogram(unit_e_all, bins=256))

        np.multiply(r, e_all, out=r)
        np.multiply(g, e_all, out=g)
        np.multiply(b, e_all, out=b)


        print('adjusted unit_image r histogram')
        print(np.histogram(r, bins=256))

        # print('final unit r histogram')
        # print(np.histogram(r, bins=256))

        # print('final unit g histogram')
        # print(np.histogram(g, bins=256))

        # print('final unit b histogram')
        # print(np.histogram(b, bins=256))

        image = ndarray_to_pil_image(unit_image)


        amax = np.amax(unit_image)
        # if amax != 0:
        byte_image = np.divide(unit_image, 255.0 / amax)

        byte_image = Image.fromarray(byte_image.astype('uint8'))
        byte_image.show()





        return

        print('unit_image histogram')
        print(np.histogram(unit_image, bins=256))

        print('unit_image CHANNEL 0 histogram')
        print(np.histogram(unit_image[:,:,0], bins=256))

        print('unit_image CHANNEL 1 histogram')
        print(np.histogram(unit_image[:,:,1], bins=256))

        print('unit_image CHANNEL 2 histogram')
        print(np.histogram(unit_image[:,:,2], bins=256))

        return





        print('stacker.stacked_image CHANNEL 0 histogram')
        print(np.histogram(stacker.stacked_image[:,:,0], bins=256))

        print('stacker.stacked_image CHANNEL 1 histogram')
        print(np.histogram(stacker.stacked_image[:,:,1], bins=256))

        print('stacker.stacked_image CHANNEL 2 histogram')
        print(np.histogram(stacker.stacked_image[:,:,2], bins=256))

        byte_image = stacker.equalize_hist(nbins=256)

        print('byte_image histogram')
        print(np.histogram(byte_image, bins=256))

        byte_image = Image.fromarray(byte_image.astype('uint8'))
        # byte_image.show()

        afn = output_pattern % n
        directory = os.path.dirname(afn)
        if directory not in dirs:
            os.makedirs(directory)
            dirs[directory] = True
        byte_image.save(afn)
        print('      =>', afn)

        return  ##############################################################
    return


def main():
    parser = argparse.ArgumentParser(
        description='Test stacking of JPEG images (all the same size and alignment). '
        'A stacked image is output for every consecutive sequence of at least MIN_DEPTH and at most DEPTH images.'
    )
    parser.add_argument(
        '--input_glob',
        required=True,
        type=str,
        help=
        'Glob pattern to be used for locating files to be processed. To avoid the shell from expanding, quote the pattern.'
    )
    parser.add_argument(
        '--output_pattern',
        required=True,
        type=str,
        help='Output file name pattern, which must include exactly one %%0Nd (where N is an integer) printf pattern.')
    parser.add_argument(
        '--depth',
        default=20,
        type=int,
        help=
        'Depth of the stack. An image is output for each consecutive sequence of DEPTH images. Must be greater than 1.'
    )
    parser.add_argument(
        '--min_depth',
        default=None,
        type=int,
        help=
        'If provided and below DEPTH, then images are output at the start and end of the sequence of input images.'
    )
    # parser.add_argument('--fps',
    #     default=5.0,
    #     type=float,
    #          help='Frames per second.')
    # parser.add_argument('--save_dir',
    #     default='',
    #     help='Directory into which to save the parts. If not set, then not saved.')
    args = parser.parse_args()

    def arg_error(msg):
        print(file=sys.stderr)
        print(msg, file=sys.stderr)
        print(file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    print(f'args={args}')

    if args.min_depth is None:
        args.min_depth = args.depth
    elif args.min_depth <= 0:
        args.min_depth = 1
    elif args.min_depth > args.depth:
        arg_error('MIN_DEPTH ({args.min_depth}) is greater than DEPTH ({args.depth}).')

    if not re.search(r'%0[1-9]\d*d', args.output_pattern):
        arg_error(f'OUTPUT_PATTERN ({args.output_pattern!r}) is invalid: required printf pattern is missing.')

    if args.output_pattern.count('%') != 1:
        arg_error(f'OUTPUT_PATTERN ({args.output_pattern!r}) is invalid: too many printf patterns.')

    image_paths = glob.glob(args.input_glob)
    if len(image_paths) < args.min_depth:
        arg_error(f'There are only {len(image_paths)} images, but need at least {args.min_depth}')

    process_images(image_paths, args.output_pattern, args.min_depth, args.depth)
    sys.exit()


if __name__ == '__main__':
    main()
    sys.exit()

    stacked_image = None

    for n in range(48):
        fn = '/home/james/Pictures/foscam-snapshots/snapshot_%d.jpg' % (n + 1)
        print('Loading', fn)
        with open(fn, 'rb') as f:
            image = jpeg_bytes_to_ndarray(f.read())
        if stacked_image is None:
            # Convert image to uint32 so there is more range available.
            stacked_image = image.astype('uint32')
        else:
            np.add(stacked_image, image, out=stacked_image)

    print('stacked_image histogram')
    print(np.histogram(stacked_image, bins=256))

    # Subtract the minimum value so that the values start at zero (black).

    zero_image = stacked_image - np.amin(stacked_image)

    print('zero_image histogram')
    print(np.histogram(zero_image, bins=256))

    # Now divide by the maximum value / 255, bringing the max value to 255.

    zero_image = zero_image.astype('double')
    byte_image = zero_image / (np.amax(zero_image) / 255.0)

    print('byte_image histogram')
    print(np.histogram(byte_image, bins=256))

    byte_image = Image.fromarray(byte_image.astype('uint8'))

    byte_image.show()

    byte_image.save('/home/james/Pictures/foscam-snapshots/average-snapshot.jpg')
