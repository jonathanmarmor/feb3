#!/usr/bin/env python

import random
import datetime
import csv
import os
from collections import Counter
import argparse

import yaml


MAX_DEPTH = 99


def try_f(f, args=[], kwargs={}, depth=0):
    """Dumb way to try a random process a bunch of times."""
    depth += 1
    try:
        return f(*args, **kwargs)
    except Exception as e:
        if depth == MAX_DEPTH:
            print "C'mon, you tried {} {} times. Fix the code already.".format(f.__name__, MAX_DEPTH)
            raise e
        return try_f(f, args=args, kwargs=kwargs, depth=depth)


def weighted_choice_lists(options, weights):
    """Choose an item from options using weights

    >>> weighted_choice_lists([1, 2], [100000, 0.000001])
    1

    """
    sum_of_weights = sum(weights)
    rand = random.uniform(0, sum_of_weights)
    total = 0
    for item, weight in zip(options, weights):
        total += weight
        if rand < total:
            return item


def weighted_choice(pairs):
    """Choose an item from a list of (item, weight) pairs

    >>> pairs = [(1, 10000), (2, 0.000001)]
    >>> weighted_choice(pairs)
    1

    """
    options, weights = zip(*pairs)
    return weighted_choice_lists(weights, options)


def weighted_choice_dict(d):
    """Choose a key from a dict using the values as weights.

    Works for collections.Counter using the counts as weights.

    >>> chords = {(0, 4, 7): 10000, (0, 1, 2): 0.000001}
    >>> weighted_choice_dict(chords)
    (0, 4, 7)

    """
    return weighted_choice(d.items())


def zero(root, chord):
    """Give the pitch classes of `chord` as if `root` was 0."""
    return tuple(sorted([(p - root) % 12 for p in chord]))


def get_all_transpositions(chord):
    return list(set([zero(p, chord) for p in chord]))


allowed_chord_types = [
    # (0,),
    # (0, 5),
    # (0, 4),
    # (0, 4, 7),
    # (0, 3, 7),
    # (0, 3),
    # (0, 5, 7),
    # (0, 3, 5),
    # (0, 2),
    # (0, 2, 5),
    # (0, 3, 7, 10),
    # (0, 4, 7, 10),
    # (0, 2, 4, 7),
    # (0, 3, 5, 7),
    # (0, 2, 5, 7),
    # (0, 2, 4, 7, 9),
    # (0, 4, 7, 11),
    # (0, 2, 4, 7, 10),
    # (0, 3, 7, 9),
    # (0, 2, 4),
    # (0, 2, 6),
    # (0, 3, 6),
    # (0, 4, 8)

    # Billboard / McGill chord types
    (0, 2, 3, 7, 9),
    (0, 4, 7, 9, 11),
    (0, 7),
    (0, 5, 7),
    (0, 4, 6, 7, 11),
    (0, 10),
    (0, 4, 5, 7, 9),
    (0, 7, 10),
    (0, 3, 4, 7, 10),
    (0, 7, 8),
    (0, 2, 5, 7, 9, 10),
    (0, 2, 5),
    (0, 4, 8),
    (0, 2, 7),
    (0, 4, 7, 9, 10),
    (0, 3, 7, 8, 10),
    (0,),
    (0, 3, 6, 9),
    (0, 4, 6, 7),
    (0, 3, 6, 10),
    (0, 4, 6, 10),
    (0, 1, 3, 6),
    (0, 1, 4, 7, 10),
    (0, 3, 5, 7),
    (0, 2, 3, 7, 11),
    (0, 2, 4, 8, 10),
    (0, 4, 6, 7, 9),
    (0, 4, 7, 11),
    (0, 5, 7, 8, 10),
    (0, 5, 7, 11),
    (0, 4, 7, 8),
    (0, 4, 5, 7, 11),
    (0, 3, 7, 9),
    (0, 2, 4, 7, 10),
    (0, 1, 4, 7, 9),
    (0, 2, 3, 7, 8, 10),
    (0, 2, 3, 5, 7, 10),
    (0, 3, 7, 8),
    (0, 3, 7, 11),
    (0, 5),
    (0, 2, 4, 6, 7),
    (0, 3, 7, 9, 10),
    (0, 8),
    (0, 3, 7),
    (0, 2, 3, 7, 10),
    (0, 2, 5, 7),
    (0, 4, 5, 7, 10),
    (0, 2, 5, 7, 11),
    (0, 4, 7, 10, 11),
    (0, 5, 6),
    (0, 2, 4, 7, 11),
    (0, 3, 5, 7, 10),
    (0, 3, 6, 11),
    (0, 4, 11),
    (0, 3, 7, 9, 11),
    (0, 4, 6, 7, 9, 11),
    (0, 3, 5, 6, 10),
    (0, 2, 5, 6, 7, 10),
    (0, 7, 9),
    (0, 2, 4, 6, 7, 11),
    (0, 2, 4),
    (0, 4, 8, 11),
    (0, 2, 5, 7, 10),
    (0, 2, 6, 7),
    (0, 4),
    (0, 2, 4, 8),
    (0, 2, 4, 6, 7, 10),
    (0, 2, 4, 5, 7, 9, 11),
    (0, 4, 7, 9),
    (0, 4, 5, 7, 9, 10),
    (0, 2, 3, 5, 10),
    (0, 4, 7, 8, 10),
    (0, 3, 5, 7, 11),
    (0, 4, 5, 7),
    (0, 1, 3, 7, 10),
    (0, 1, 4, 7),
    (0, 1, 4, 5, 7, 10),
    (0, 2, 4, 6, 7, 9, 10),
    (0, 3, 7, 10),
    (0, 4, 7),
    (0, 9),
    (0, 2, 4, 7, 9, 10),
    (0, 3, 10),
    (0, 2, 4, 7, 8, 10),
    (0, 2, 7, 10),
    (0, 2, 4, 7, 9, 11),
    (0, 3),
    (0, 2, 4, 7, 9),
    (0, 2, 3, 5, 7),
    (0, 2, 3, 6, 9),
    (0, 4, 7, 8, 11),
    (0, 2, 3, 7),
    (0, 3, 4, 7, 11),
    (0, 2, 3, 5, 7, 9, 10),
    (0, 3, 6, 8),
    (0, 3, 4, 7),
    (0, 4, 6, 7, 10),
    (0, 3, 6, 7, 9),
    (0, 2, 4, 5, 7, 10),
    (0, 2, 4, 5, 7, 9, 10),
    (0, 3, 4, 7, 8, 10),
    (0, 3, 6),
    (0, 1, 4, 7, 8, 10),
    (0, 2, 4, 7),
    (0, 2, 4, 5, 7, 11),
    (0, 1, 4, 6, 7, 10),
    (0, 4, 8, 10),
    (0, 2, 4, 6, 7, 9, 11),
    (0, 5, 7, 10),
    (0, 4, 7, 10)
]
allowed_chord_types_transpositions = []
for c in allowed_chord_types:
    allowed_chord_types_transpositions.extend(get_all_transpositions(c))
allowed_chord_types_transpositions.append(())


def is_allowed(chord):
    if not chord:
        return True
    return zero(chord[0], chord) in allowed_chord_types_transpositions


def find_supersets(subset, chord_type):
    supersets = []
    chord_type = list(chord_type)
    for offset in subset:
        for i, root in enumerate(chord_type):
            transposition = chord_type[i:] + chord_type[:i]
            transposition = tuple(sorted([(p - root + offset) % 12 for p in transposition]))
            if all([(p in transposition) for p in subset]):
                if transposition not in supersets:
                    supersets.append(transposition)
    return supersets


def find_all_supersets(subset):
    if not subset:
        subset = [random.choice(range(12))]
    supersets = []
    for chord_type in allowed_chord_types:
        supersets.extend(find_supersets(subset, chord_type))
    return supersets


note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def spell(chord):
    # TODO detect if flats or sharps should be used
    return ' '.join([note_names[p] for p in chord])


class Piece(object):
    def __init__(self, n_events=72, config='musicians.yaml'):
        self.n_events = n_events
        self.n = 0

        self.musicians = yaml.load(open(config, 'rb'))

        self.prev_state = {name: [] for name in self.musicians}
        self.prev_event = {}
        self.prev_harmony = ()

        self.score = []
        self.grid = {name: [] for name in self.musicians}

    def run(self):
        while self.n < self.n_events:
            event = try_f(self.get_event)
            # event = self.get_event()
            if event:
                self.add_event(event)
            self.n += 1

    def pick_harmony(self, entering, harmony_options, holdover_pitches):
        pitches = {name: [] for name in entering}

        if len(harmony_options) > 1 and self.prev_harmony in harmony_options:
            harmony_options.remove(self.prev_harmony)

        # pick a harmony
        print 'N Harmony Options:', len(harmony_options)
        harmony_options.reverse()
        harmony_weights = [int(2 ** n) for n in range(len(harmony_options))]
        new_harmony = weighted_choice_lists(harmony_options, harmony_weights)
        new_pitches = [p for p in new_harmony if p not in holdover_pitches]

        # make sure all new pitches are used
        n = 0
        while new_pitches:
            n += 1
            name = random.choice(entering)
            p = random.choice(new_pitches)
            if len(pitches[name]) < self.musicians[name]['max_notes']:
                pitches[name].append(p)
                new_pitches.remove(p)
            if n > 1000:
                raise Exception('Couldnt allocate all new pitches.')

        # make sure all musicians in entering get pitches
        n = 0
        while not all(pitches.values()):
            n += 1
            empty = [name for name in pitches if not pitches[name]]
            name = random.choice(empty)
            p = random.choice(new_harmony)
            pitches[name].append(p)
            if n > 1000:
                raise Exception('Couldnt fill all entering instruments.')

        # Add some extra notes
        if random.random() < 0.40:
            headroom = {name: self.musicians[name]['max_notes'] - len(pitches[name]) for name in entering}
            for name in headroom:
                pitch_options = [p for p in new_harmony if p not in pitches[name]]
                upper = min([len(pitch_options), headroom[name]])
                if upper:
                    n_pitches = 1
                    if upper > 1:
                        n_pitches = random.randint(1, upper)
                    ps = random.sample(pitch_options, n_pitches)
                    pitches[name].extend(ps)

        for name in pitches:
            pitches[name].sort()

        return pitches

    def make_new_harmony(self, entering, holdover_pitches):
        if not entering and not is_allowed(holdover_pitches):
            raise Exception('Pitches dropped out, no new pitches are coming in, and the harmony left behind is not allowed. Try again.')

        if not entering:
            return {}

        harmony_options = find_all_supersets(holdover_pitches)
        # return self.pick_harmony(entering, harmony_options, holdover_pitches)
        return try_f(self.pick_harmony, args=[entering, harmony_options, holdover_pitches])

    def get_pitches(self, changing):
        event = {}
        entering = []
        for name in changing:
            if not self.prev_state[name]:
                entering.append(name)
            else:
                event[name] = 'stop'
        # Get pitches that are sustaining from previous
        holdover_pitches = []
        not_changing = [name for name in self.musicians if name not in changing]
        holdovers = [name for name in not_changing if self.prev_state[name]]
        for name in holdovers:
            for p in self.prev_state[name]:
                if p not in holdover_pitches:
                    holdover_pitches.append(p)
        holdover_pitches.sort()

        event.update(self.make_new_harmony(entering, holdover_pitches))

        return event

    def get_event(self):
        if self.n_events - self.n < len(self.musicians):
            # End game, everyone needs to stop
            playing = [name for name in self.prev_state if self.prev_state[name]]
            if not playing:
                # We're done.
                self.n = self.n_events
                return
            if len(playing) == 1:
                changing = playing
            else:
                n_musicians_opts = range(1, len(playing) + 1)
                n_musicians_weights = list(reversed([2 ** n for n in n_musicians_opts]))
                n_musicians_weights[0] = n_musicians_weights[1]
                num_changing = weighted_choice_lists(n_musicians_opts, n_musicians_weights)
                changing = random.sample(playing, num_changing)
        else:
            not_eligible = [name for name in self.prev_event if self.prev_event[name] != 'stop']
            if len(not_eligible) == len(self.musicians):
                not_eligible.remove(random.choice(not_eligible))
            eligible = [name for name in self.musicians if name not in not_eligible]
            if len(eligible) == 1:
                changing = eligible
            else:
                n_musicians_opts = range(1, len(eligible) + 1)
                n_musicians_weights = list(reversed([2 ** n for n in n_musicians_opts]))
                n_musicians_weights[0] = n_musicians_weights[1]
                num_changing = weighted_choice_lists(n_musicians_opts, n_musicians_weights)
                changing = random.sample(eligible, num_changing)
        # return self.get_pitches(changing)
        return try_f(self.get_pitches, args=[changing])

    def add_event(self, event):
        self.score.append(event)

        self.prev_event = event

        self.prev_state = {}
        for name in event:
            self.prev_state[name] = event[name]
            if event[name] == 'stop':
                self.prev_state[name] = []

        for name in event:
            self.grid[name].append(event[name])

        not_changing = [name for name in self.musicians if name not in event]
        for name in not_changing:
            prev = []
            if self.grid[name]:
                prev = self.grid[name][-1]
            if prev == 'stop':
                prev = []
            self.grid[name].append(prev)
            self.prev_state[name] = prev

        self.prev_harmony = self.get_harmony()

    def get_harmony(self):
        pitches = []
        for name in self.grid:
            if self.grid[name][-1] and self.grid[name][-1] is not 'stop':
                for p in self.grid[name][-1]:
                    if p not in pitches:
                        pitches.append(p)
        pitches.sort()
        return tuple(pitches)

    # Reporting, displaying

    def report_score(self):
        for i, event in enumerate(self.score):
            print i
            for name in event:
                action = event[name]
                if action != 'stop':
                    action = spell(event[name])
                print '  {:>10} {}'.format(name, action)
            print

    def report_rhythm(self):
        for name in self.grid:
            line = []
            for event in self.grid[name]:
                if event == [] or event == 'stop':
                    line.append(' ')
                else:
                    line.append('-')
            print '{:<15}  {}'.format(name, ''.join(line))

    def report_harmonies(self):
        c = Counter()
        lines = []
        actual_length = len(self.grid[self.grid.keys()[0]])
        for e in range(actual_length):
            pitches = []
            for name in self.musicians:
                if self.grid[name][e] != 'stop':
                    for p in self.grid[name][e]:
                        if p not in pitches:
                            pitches.append(p)
            harmony = tuple(sorted(pitches))
            c[harmony] += 1
            line = []
            for pc in range(12):
                if pc in pitches:
                    line.append('{:<3}'.format(pc))
                else:
                    line.append('   ')
            lines.append(''.join(line))
        for line in lines:
            print line
        print
        print 'Number of different chords: ', len(c)
        for k, n in c.most_common():
            print n, k
        return lines

    def to_csv(self):
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if not os.path.isdir('exports'):
            os.mkdir('exports')
        with open('exports/two_three_four_{}.csv'.format(now), 'wb') as f:
            writer = csv.writer(f)
            for i, event in enumerate(self.score):
                line_number = i + 1
                if len(event) > 1:
                    writer.writerow([line_number, None, None])
                    line_number = None
                for name in event:
                    action = event[name]
                    if action == 'stop':
                        action = None
                    else:
                        action = spell(event[name])
                    writer.writerow([line_number, name, action])

    def pitches_in_part(self, name):
        pitches = set()
        for event in self.grid[name]:
            if event and event != 'stop':
                pitches.update(event)
        pitches = list(pitches)
        pitches.sort()
        return pitches

    def reports(self):
        print
        self.report_score()
        print
        self.report_rhythm()
        print
        self.report_harmonies()
        print
        # connor = self.pitches_in_part('Connor')
        # print 'connor', connor
        # print 'len(connor)', len(connor)
        # return self


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='musicians.yaml', help='Config file defining the musicians.')
    parser.add_argument('--events', '-e', default=72, help='The number of events to make.', type=int)
    args = parser.parse_args()

    p = Piece(n_events=args.events, config=args.config)
    # p.test()
    p.run()
    p.reports()
