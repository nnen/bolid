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
    def __init__(self, data, **kwargs):
        self.data = data
        self.mode = kwargs.get("mode", "L")
        
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

    def get_avg(self):
        return float(sum(self.data)) / float(len(self.data))
    
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
            result[bucket] = float(total) / float(width)
        
        total = 0
        width = 0
        for count in e:
            total += count
            width += 1
        result[-1] = float(total) / float(width)
        
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
    
    def plot(self, width = 200, height = 100, step = 4):
        img = Image.new("RGBA", (width, height), (255, 255, 255, 50))
        hist = self.discretize(width / step)
        m = min(float(hist.get_max()), float(self.get_avg() + 2.0 * self.get_variance()))
        norm = lambda c: min(height, int(float(10 * height) * (float(c) / m)))
        data = img.load()
        for color, count in enumerate(hist.data):
            for y in range(norm(count)):
                for x in range(step):
                    data[(color * step) + x, height - y - 1] = (255, 0, 0, 130)
        return img
    
    @classmethod
    def from_image_hist(cls, img, band = None):
        if img.mode == "RGBA":
            if band is None:
                return cls(img.histogram()[:256 * 3], mode = img.mode)
            else:
                from_ = 256 * band
                to = from_ + 256
                return cls(img.histogram()[from_:to], mode = img.mode)
        if img.mode == "RGB":
            if band is None:
                return cls(img.histogram()[256:512], mode = img.mode)
            else:
                from_ = 256 * band
                to = from_ + 256
                return cls(img.histogram()[from_:to], mode = img.mode)
        return cls(img.histogram(), mode = img.mode)
    
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


def enum(*args, **kwargs):
    enums = dict(zip(args, range(len(args))), **kwargs)
    return type("Enum", (), enums)


Classes = enum("BOLID", "ACTIVITY", "NONE")


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
            to_save = img.convert("RGBA")
            hist = Histogram.from_image_hist(to_save, 1).plot(width = self.box[2] - self.box[0], step = 4)
            w, h = hist.size
            to_save.paste(hist, (0, 0, w, h), hist)
            to_save.save(filename)
        return img
    
    def detect(self, img, filename = None):
        img = self.filter_image(img, filename)
        hist = Histogram.from_image_hist(img)
        hist = hist.discretize(buckets = self.bucket_count)
        hist = hist.normalize()
        
        value = sum([hist[i] for i in self.buckets])
        if value < self.threshold:
            return Classes.NONE, hist
        
        data = img.load()
        w, h = img.size
        
        rows = []
        for y in range(0, h, 3):
            s = 0
            for x in range(0, w, 4):
                s += data[x, y]
            rows.append(s)
        
        thr = max(rows) / 2
        count = 0
        for row in rows:
            if row > thr: count += 1
        
        if count < 5:
            return Classes.ACTIVITY, hist
        
        return Classes.BOLID, hist


def main():
    parser = optparse.OptionParser()
    parser.add_option("-v", "--verbose", dest = "verbose",
                      action = "store_true", default = False,
                      help = "print out more information")
    parser.add_option("-f", "--filtered", dest = "filtered",
                      action = "store_true", default = False,
                      help = "save filtered images")
    parser.add_option("-i", "--inverted", dest = "inverted",
                      action = "store_true", default = False,
                      help = "invert the search - printout files with no bolid")
    options, args = parser.parse_args()
    
    for fn in args:
        if fn.endswith("_filtered.jpg"): continue
        img = Image.open(fn)
        if img.mode != "RGB": continue
        
        detector = BolidDetector(filename = fn, save_filtered = options.filtered)
        activity, hist = detector.detect(img, fn)
        
        if options.verbose:
            print "%s\t%s\t%s" % (
                fn,
                {
                    Classes.BOLID: "B",
                    Classes.ACTIVITY: "a",
                    Classes.NONE: " ",
                }.get(activity, " "),
                "\t".join(map(lambda c: "%0.4f" % (c, ), hist)),
            )
        elif options.inverted:
            if activity != Classes.BOLID:
                print fn
        elif activity == Classes.BOLID:
            print fn


if __name__ == "__main__":
    main()

