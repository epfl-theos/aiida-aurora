# -*- coding: utf-8 -*-
"""
Parsers provided by aiida_aurora.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""
from aiida.engine import ExitCode
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory
from aiida.common import exceptions
# from aiida.orm import SinglefileData
from aiida.orm import Dict
import json

DiffCalculation = CalculationFactory('aurora')


class DiffParser(Parser):
    """
    Parser class for parsing output of calculation.
    """

    def __init__(self, node):
        """
        Initialize Parser instance

        Checks that the ProcessNode being passed was produced by a DiffCalculation.

        :param node: ProcessNode of calculation
        :param type node: :class:`aiida.orm.ProcessNode`
        """
        super().__init__(node)
        if not issubclass(node.process_class, DiffCalculation):
            raise exceptions.ParsingError('Can only parse DiffCalculation')

    def parse(self, **kwargs):
        """
        Parse outputs, store results in database.

        :returns: an exit code, if parsing fails (or nothing if parsing succeeds)
        """
        stdout_filename = self.node.get_option('output_filename')
        output_json_filename = self.node._OUTPUT_JSON_FILE

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()
        files_expected = [stdout_filename, output_json_filename]
        # Note: set(A) <= set(B) checks whether A is a subset of B
        if not set(files_expected) <= set(files_retrieved):
            self.logger.error("Found files '{}', expected to find '{}'".format(files_retrieved, files_expected))
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # add output file
        self.logger.info("Parsing '{}'".format(output_json_filename))
        with self.retrieved.open(output_json_filename, 'r') as handle:
            output_node = Dict(dict=json.load(handle))  # this should be changed in the appropriate node data type
        self.out('results', output_node)

        return ExitCode(0)
