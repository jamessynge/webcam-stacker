#!/usr/bin/env python3.6
"""
Initial experiment in stacking a set of captured snapshots.
"""

import argparse
import glob
import io
import logging
import numpy as np
import os
from PIL import Image
import re
import sys
from skimage import exposure


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


class ImageStacker(object):
    """docstring for ImageStacker"""

    def __init__(self):
        super(ImageStacker, self).__init__()
        self.stacked_image = None
        self.images = []

    def num_images(self):
        return len(self.images)

    def add(self, image):
        check_types(image, np.ndarray)
        self.images.append(image)
        if self.stacked_image is None:
            self.stacked_image = image.astype('double')  # Maybe copy it.
        else:
            check_equal(image.shape, self.stacked_image.shape)
            np.add(self.stacked_image, image, out=self.stacked_image)

    def remove_oldest(self):
        oldest = self.images.pop(0)
        np.add(self.stacked_image, oldest, out=self.stacked_image)
        return oldest

    def scale(self, scale):
        np.multiply(self.stacked_image, scale, out=self.stacked_image)

    def get_unit_image(self):
        zero_image = self.stacked_image - np.amin(self.stacked_image)
        unit_image = zero_image / np.amax(zero_image)
        return unit_image

    def get_byte_image(self):
        # print('stacked_image histogram')
        # print(np.histogram(self.stacked_image, bins=256))

        # Subtract the minimum value so that the values start at zero (black).

        zero_image = self.stacked_image - np.amin(self.stacked_image)

        # print('zero_image histogram')
        # print(np.histogram(zero_image, bins=256))

        # Now divide by the maximum value / 255, bringing the max value to 255.
        zero_image = zero_image.astype('double')
        byte_image = zero_image / (np.amax(zero_image) / 255.0)

        # print('byte_image histogram')
        # print(np.histogram(byte_image, bins=256))

        return byte_image

    def equalize_hist(self, nbins=256):
        print('stacked_image shape:', self.stacked_image.shape)
        print('stacked_image amin:', np.amin(self.stacked_image))
        print('stacked_image amax:', np.amax(self.stacked_image))

        image = exposure.equalize_hist(
            self.stacked_image, nbins=nbins)
        print('equalize_hist image shape:', image.shape)
        print('equalize_hist image amin:', np.amin(image))
        print('equalize_hist image amax:', np.amax(image))

        byte_image = image / (np.amax(image) / 255.0)
        return byte_image

    def equalize_adapthist(self, kernel_size=None, clip_limit=0.01, nbins=256):
        print('stacked_image shape:', self.stacked_image.shape)
        print('stacked_image amin:', np.amin(self.stacked_image))
        print('stacked_image amax:', np.amax(self.stacked_image))

        image = exposure.equalize_adapthist(
            self.stacked_image, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins)
        print('adapthist image shape:', image.shape)
        print('adapthist image amin:', np.amin(image))
        print('adapthist image amax:', np.amax(image))

        byte_image = image / (np.amax(image) / 255.0)
        return byte_image


def process_images(image_paths, output_pattern, min_depth, depth):
    image_paths.sort()
    stacker = ImageStacker()
    dirs = {}
    for n, fn in enumerate(image_paths):
        print('Loading', fn)
        with open(fn, 'rb') as f:
            image = jpeg_bytes_to_ndarray(f.read())
        stacker.add(image)

        while stacker.num_images() > depth:
            stacker.remove_oldest()

        if stacker.num_images() < min_depth:
            continue


        # print('stacker.stacked_image histogram')
        # print(np.histogram(stacker.stacked_image, bins=256))

        unit_image = stacker.get_unit_image()
        r = unit_image[:,:,0]
        g = unit_image[:,:,1]
        b = unit_image[:,:,2]

        # print('unit_image r histogram')
        # print(np.histogram(r, bins=256))

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



        # print('final unit r histogram')
        # print(np.histogram(r, bins=256))

        # print('final unit g histogram')
        # print(np.histogram(g, bins=256))

        # print('final unit b histogram')
        # print(np.histogram(b, bins=256))

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
