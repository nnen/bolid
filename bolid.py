#!/usr/bin/python
# -*- coding: utf-8 -*-
"""bolid module.

Author: Jan Milík <milikjan@fit.cvut.cz>
"""


import sys
import optparse

import Image
import ImageFilter


class Histogram(object):
    def __init__(self, data):
        self.data = data
        
        total = 0
        for value in data:
            total += value
        self.total = total
        #self.maxv = max(self.data)
        #self.minv = min(self.data)
    
    def __getitem__(self, key):
        return self.data.__getitem__(key)
    
    def __iter__(self):
        return iter(self.data)
    
    def get_total(self):
        return sum(self.data)
    
    def get_max(self):
        return max(self.data)
    
    def discretize(self, buckets = 20):
        data = self.data
        
        if buckets > len(data):
            return self
        
        width = int(float(len(data)) / float(buckets))
        result = [0, ] * buckets
        
        e = iter(data)
        
        for bucket in range(buckets - 1):
            total = 0
            for w in range(width):
                count = next(e)
                total += count
            result[bucket] = total
        
        total = 0
        for count in e:
            total += count
        result[-1] = total
        
        return Histogram(result)
    
    def normalize(self):
        data = self.data
        total = self.total
        
        if float(total) == 0.0:
            new_data = [0.0, ] * len(data)
        else:
            new_data = map(lambda c: float(c) / float(total), data)
        
        return Histogram(new_data)
    
    def remove(self, bucket):
        self.data.pop(bucket)
        self.total = self.get_total()
    
    def remove_range(self, bucket_from, bucket_to):
        data = self.data
        new_data = data[:bucket_from] + data[bucket_to:]
        return Histogram(new_data)
    
    def print_ascii(self, width = 80, out = sys.stdout, norm = None):
        if norm is None:
            m = self.get_total()
            norm = lambda v: int(float(width) * (float(v) / float(m)))
        
        if isinstance(self.data[0], int):
            count_fn = lambda c: "  %d" % (c, )
        else:
            count_fn = lambda c: "  %.4f" % (c, )
        
        print "      |%s|" % ("-" * 78, )
        
        for bucket, v in enumerate(self.data):
            out.write("%4d  " % (bucket, ))
            out.write("*" * norm(v))
            #out.write("  %r  %.4f" % (v, float(v) / float(self.total)))
            out.write(count_fn(v))
            out.write("\n")
        
        return norm
    
    def get_average(self):
        result = 0.0
        for value, count in enumerate(self.data):
            result += float(value) * float(count)
        result = result / float(self.total)
        return result
    
    def get_variance(self):
        ex = self.get_average()
        ex2 = 0.0
        for value, count in enumerate(self.data):
            ex2 += float(count) * (float(value * value))
        ex2 = ex2 / float(self.total)
        return ex2 - (ex * ex)
    
    @classmethod
    def from_image_hist(cls, img):
        if img.mode == "RGB":
            return cls(img.histogram()[256:512])
        return cls(img.histogram())
    
    @classmethod
    def from_colors(cls, colors, channel = 1):
        data = []
        for count, color in colors:
            ch = color[channel]
            
            if ch >= len(data):
                data.extend([0, ] * (ch - len(data) + 1))
            
            data[ch] += count
        
        return cls(data)
    
    @classmethod
    def from_image(cls, img):
        w, h = img.size
        colors = img.getcolors(w * h)
        return cls.from_colors(colors)


class BolidDetector(object):
    def __init__(self, **kwargs):
        self.save_filtered = kwargs.get("save_filtered", False)
        
        self.box = (400, 40, 600, 500)
        self.bucket_count = 3
        self.buckets = [-1, -2]
        self.threshold = 0.001
    
    def filter_image(self, img, filename = None):
        assert img.mode == "RGB"
        img = img.crop(self.box)
        img = img.split()[1]
        img = img.filter(ImageFilter.MedianFilter(5))
        if self.save_filtered and filename:
            filename = filename.split(".")[0] + "_filtered.jpg"
            img.save(filename)
        return img
    
    def detect(self, img, filename = None):
        img = self.filter_image(img, filename)
        hist = Histogram.from_image_hist(img)
        hist = hist.normalize()
        hist = hist.discretize(buckets = self.bucket_count)
        
        value = sum([hist[i] for i in self.buckets])
        return value > self.threshold, hist


def filter_img(img, hist = None):
    assert img.mode == "RGB"
    img = img.split()[1]
    
    return img.filter(ImageFilter.MedianFilter(5))


def detect_activity(hist):
    #hist = hist.remove_range(0, 50)
    hist = hist.normalize()
    hist = hist.discretize(buckets = 3)
    
    value = hist[-1] + hist[-2]
    return value > 0.001, hist


def main():
    parser = optparse.OptionParser()
    parser.add_option("-v", "--verbose", dest = "verbose",
                      action = "store_true", default = False,
                      help = "print out more information")
    #parser.add_option("-s", "--show", dest = "show",
    #                  action = "store_true", default = False,
    #                  help = "show images with detected activity")
    parser.add_option("-f", "--filtered", dest = "filtered",
                      action = "store_true", default = False,
                      help = "save filtered images")
    options, args = parser.parse_args()
    
    norm = None
    for fn in args:
        img = Image.open(fn)
        if img.mode != "RGB": continue
        
        detector = BolidDetector(filename = fn, save_filtered = options.filtered)
        activity, hist = detector.detect(img, fn)
        
        if options.verbose:
            print "%s\t%s\t%s" % (
                fn,
                "b" if activity else "-",
                "\t".join(map(lambda c: "%0.4f" % (c, ), hist)),
            )
        elif activity:
            print fn


if __name__ == "__main__":
    main()

