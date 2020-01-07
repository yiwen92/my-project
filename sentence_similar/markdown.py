from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import logging
'''
from training_data import Message, TrainingData
from training_data.util import check_duplicate_synonym
from utils import build_entity

from training_data.formats.readerwriter import TrainingDataReader, TrainingDataWriter
'''
INTENT = "intent"
SYNONYM = "synonym"
REGEX = "regex"
LOOKUP = "lookup"
available_sections = [INTENT, SYNONYM, REGEX, LOOKUP]
ent_regex = re.compile(r'\[(?P<entity_text>[^\]]+)'
                       r'\]\((?P<entity>[^:)]*?)'
                       r'(?:\:(?P<value>[^)]+))?\)')  # [entity_text](entity_type(:entity_synonym)?)

item_regex = re.compile(r'\s*[-\*+]\s*(.+)')
comment_regex = re.compile(r'<!--[\s\S]*?--!*>', re.MULTILINE)
fname_regex = re.compile(r'\s*([^-\*+]+)')

logger = logging.getLogger(__name__)


class MarkdownReader(TrainingDataReader):
    """Reads markdown training data and creates a TrainingData object."""

    def __init__(self):
        self.current_title = None
        self.current_section = None
        self.training_examples = []
        self.entity_synonyms = {}
        self.regex_features = []
        self.section_regexes = self._create_section_regexes(available_sections)
        self.lookup_tables = []

    def reads(self, s, **kwargs):
        """Read markdown string and create TrainingData object"""
        self.__init__()
        s = self._strip_comments(s)
        for line in s.splitlines():
            line = line.strip()
            header = self._find_section_header(line)
            if header:
                self._set_current_section(header[0], header[1])
            else:
                self._parse_item(line)
                self._load_files(line)
        return TrainingData(self.training_examples, self.entity_synonyms,
                            self.regex_features, self.lookup_tables)

    @staticmethod
    def _strip_comments(text):
        """ Removes comments defined by `comment_regex` from `text`. """
        return re.sub(comment_regex, '', text)

    @staticmethod
    def _create_section_regexes(section_names):
        def make_regex(section_name):
            return re.compile(r'##\s*{}:(.+)'.format(section_name))

        return {sn: make_regex(sn) for sn in section_names}

    def _find_section_header(self, line):
        """Checks if the current line contains a section header and returns the section and the title."""
        for name, regex in self.section_regexes.items():
            match = re.search(regex, line)
            if match is not None:
                return name, match.group(1)
        return None

    def _load_files(self, line):
        """Checks line to see if filename was supplied.  If so, inserts the
        filename into the lookup table slot for processing from the regex
        featurizer."""
        if self.current_section == LOOKUP:
            match = re.match(fname_regex, line)
            if match:
                fname = match.group(1)
                self.lookup_tables.append(
                    {"name": self.current_title, "elements": str(fname)})

    def _parse_item(self, line):
        """Parses an md list item line based on the current section type."""
        match = re.match(item_regex, line)
        if match:
            item = match.group(1)
            if self.current_section == INTENT:
                parsed = self._parse_training_example(item)
                self.training_examples.append(parsed)
            elif self.current_section == SYNONYM:
                self._add_synonym(item, self.current_title)
            elif self.current_section == REGEX:
                self.regex_features.append(
                    {"name": self.current_title, "pattern": item})
            elif self.current_section == LOOKUP:
                self._add_item_to_lookup(item)

    def _add_synonym(self, text, value):
        check_duplicate_synonym(self.entity_synonyms, text, value, "reading markdown")
        self.entity_synonyms[text] = value

    def _add_synonyms(self, plain_text, entities):
        """Adds synonyms found in intent examples"""
        for e in entities:
            e_text = plain_text[e['start']:e['end']]
            if e_text != e['value']:
                self._add_synonym(e_text, e['value'])

    def _parse_training_example(self, example):
        """Extract entities and synonyms, and convert to plain text."""
        entities = self._find_entities_in_training_example(example)
        plain_text = re.sub(ent_regex, lambda m: m.groupdict()['entity_text'], example)
        self._add_synonyms(plain_text, entities)
        message = Message(plain_text, {'intent': self.current_title})
        if len(entities) > 0:
            message.set('entities', entities)
        return message

    def _set_current_section(self, section, title):
        """Update parsing mode."""
        if section not in available_sections:
            raise ValueError("Found markdown section {} which is not "
                             "in the allowed sections {},".format(section, ",".join(available_sections)))

        self.current_section = section
        self.current_title = title


class MarkdownWriter(TrainingDataWriter):

    def dumps(self, training_data):
        """Transforms a TrainingData object into a markdown string."""
        md = u''
        md += self._generate_training_examples_md(training_data)
        md += self._generate_synonyms_md(training_data)
        md += self._generate_regex_features_md(training_data)
        md += self._generate_lookup_tables_md(training_data)

        return md

    def _generate_training_examples_md(self, training_data):
        """generates markdown training examples."""
        training_examples = sorted([e.as_dict() for e in training_data.training_examples],
                                   key=lambda k: k['intent'])
        md = u''
        for i, example in enumerate(training_examples):
            if i == 0 or training_examples[i - 1]['intent'] != example['intent']:
                md += self._generate_section_header_md(INTENT, example['intent'], i != 0)

            md += self._generate_item_md(self._generate_message_md(example))

        return md

    def _generate_section_header_md(self, section_type, title, prepend_newline=True):
        """generates markdown section header."""
        prefix = "\n" if prepend_newline else ""
        return prefix + "## {}:{}\n".format(section_type, title)

    def _generate_item_md(self, text):
        """generates markdown for a list item."""
        return "- {}\n".format(text)

    def _generate_fname_md(self, text):
        """generates markdown for a lookup table file path."""
        return "  {}\n".format(text)
