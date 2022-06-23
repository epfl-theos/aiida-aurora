# -*- coding: utf-8 -*-
"""
Plugin for the Tomato scheduler.
"""
import re
import logging
import yaml

from aiida.common.escaping import escape_for_bash
from aiida.schedulers import Scheduler, SchedulerError
from aiida.schedulers.datastructures import JobInfo, JobResource, JobState
from aiida.common.exceptions import FeatureNotAvailable
from aiida.common.extendeddicts import AttributeDict

_LOGGER = logging.getLogger(__name__)

# There is no "job owner" in tomato.

# Mapping of Tomato states to AiiDA `JobState`s
## The following statuses are defined in tomato:
## q  job is queued. Jobs shouldn't stay in q too long as that indicates there
##      is no pipeline that can process the payload.
## qw job is queued and a matching pipeline has been found, but it is either busy,
##      not ready, or without the correct sample
## r  job is running
## c  job has completed successfully
## ce job has completed with an error - output data not guaranteed, might be present in the job folder.
## cd job has been cancelled - output data should be available as specified in the yamlfile

_MAP_STATUS_TOMATO = {
    'q': JobState.QUEUED,  # JobState.QUEUED_HELD ?
    'qw': JobState.QUEUED,
    'r': JobState.RUNNING,
    'c': JobState.DONE,
    'ce': JobState.DONE,
    'cd': JobState.DONE,
}

# ketchup submit returns "jobid = {jobid}"
_TOMATO_SUBMITTED_REGEXP = re.compile(r'(.*:\s*)?(jobid\s+=)\s+(?P<jobid>\d+)')


class TomatoResource(JobResource):
    """Class for Tomato job resources."""

    @classmethod
    def validate_resources(cls, **kwargs):
        """Validate the resources against the job resource class of this scheduler.

        :param kwargs: dictionary of values to define the job resources
        :raises ValueError: if the resources are invalid or incomplete
        :return: optional tuple of parsed resource settings
        """
        resources = AttributeDict()
        return resources

    @classmethod
    def accepts_default_mpiprocs_per_machine(cls):
        """Return True if this subclass accepts a `default_mpiprocs_per_machine` key, False otherwise."""
        return False

    def get_tot_num_mpiprocs(self):
        """Return the total number of cpus of this job resource."""
        return 1


class TomatoScheduler(Scheduler):
    """
    Support fot the Tomato scheduler (https://github.com/dgbowl/tomato)
    """

    # Query only by list of jobs and not by user
    _features = {
        'can_query_by_user': False,
    }

    # The class to be used for the job resource.
    _job_resource_class = TomatoResource

    _map_status = _MAP_STATUS_TOMATO

    def _get_joblist_command(self, jobs=None, user=None):
        """The command to report full information on existing jobs.

        :return: a string of the command to be executed to determine the active jobs.
        """

        command = ['ketchup', '-t', 'status']

        if user:
            raise FeatureNotAvailable('Cannot query by user')

        if jobs:
            if len(jobs) > 1:
                raise FeatureNotAvailable('ketchup status supports only one job per time')

            if isinstance(jobs, str):
                command.append(f'{escape_for_bash(jobs)}')
            else:
                try:
                    command.append(f"{' '.join(escape_for_bash(j) for j in jobs)}")
                except TypeError:
                    raise TypeError("If provided, the 'jobs' variable must be a string or an iterable of strings")
        else:
            command.append('queue')

        comm = ' '.join(command)
        _LOGGER.debug(f'ketchup command: {comm}')
        return comm

    def _get_detailed_job_info_command(self, job_id):
        """Return the command to run to get the detailed information on a job,
        even after the job has finished.

        The output text is just retrieved, and returned for logging purposes.
        """
        return f'ketchup -t status {escape_for_bash(job_id)}'

    def _get_submit_script_header(self, job_tmpl):
        """Return the submit script final part, using the parameters from the job template.

        :param job_tmpl: a ``JobTemplate`` instance with relevant parameters set.
        """
        lines = []

        # TODO: read the payload version and load the appropriate schema
        # from dgbowl_schemas.tomato.payload_0_1.tomato import Tomato
        # - should 'version: "0.1"' be written in the submit script?
        # - name of the file containing the sample and method
        # PAYLOAD v 0.1 has everything in one file... but tomato and sample/method should be divided

        if job_tmpl.tomato_schema:
            lines.append(yaml.dump({'tomato': job_tmpl.tomato_schema.dict()}))
        else:
            raise ValueError('tomato_schema is required for the Tomato scheduler plugin')

        return '\n'.join(lines)

    def _get_submit_command(self, submit_script):
        """Return the string to execute to submit a given script.

        .. warning:: the `submit_script` should already have been bash-escaped

        :param submit_script: the path of the submit script relative to the working directory.
        :return: the string to execute to submit a given script.
        """
        submit_command = f'ketchup -t submit {submit_script}'

        _LOGGER.info(f'submitting with: {submit_command}')

        return submit_command

    def _parse_joblist_output(self, retval, stdout, stderr):
        """Parse the joblist output as returned by executing the command returned by `_get_joblist_command` method.

        :return: list of `JobInfo` objects, one of each job each with at least its default params implemented.
        """
        if retval != 0:
            raise SchedulerError(
                f"""kethup returned exit code {retval} (_parse_joblist_output function)"""
                f"""stdout='{stdout.strip()}'"""
                f"""stderr='{stderr.strip()}'"""
            )
        if stderr.strip():
            self.logger.warning(
                f"ketchup returned exit code 0 (_parse_joblist_output function) but non-empty stderr='{stderr.strip()}'"
            )

        jobdata_raw = [
            l.split()
            for l in stdout.splitlines()
            if l and 'jobid' not in l and '==========================================' not in l
        ]

        # Create dictionary and parse specific fields
        job_list = []
        for job in jobdata_raw:

            this_job = JobInfo()
            this_job.job_id = job[0]

            try:
                this_job.job_state = _MAP_STATUS_TOMATO[job[1]]
            except KeyError:
                self.logger.warning(f"Unrecognized job_state '{job[1]}' for job id {this_job.job_id}")
                this_job.job_state = JobState.UNDETERMINED

            if len(job) == 2:
                this_job.pipeline = None
            elif len(job) == 3:
                this_job.pipeline = job[2]
            else:
                raise ValueError('More than 3 columns returned by ketchup -t status queue')

            # Everything goes here anyway for debugging purposes
            this_job.raw_data = job

            # I append to the list of jobs to return
            job_list.append(this_job)

        return job_list

    def _parse_submit_output(self, retval, stdout, stderr):
        """Parse the output of the submit command returned by calling the `_get_submit_command` command.

        :return: a string with the job ID.
        """
        if retval != 0:
            _LOGGER.error(f'Error in _parse_submit_output: retval={retval}; stdout={stdout}; stderr={stderr}')
            raise SchedulerError(f'Error during submission, retval={retval}; stdout={stdout}; stderr={stderr}')

        if stderr.strip():
            _LOGGER.warning(f'in _parse_submit_output there was some text in stderr: {stderr}')

        # I check for a valid string in the output.
        # See comments near the regexp above.
        # I check for the first line that matches.
        for line in stdout.split('\n'):
            match = _TOMATO_SUBMITTED_REGEXP.match(line.strip())
            if match:
                return match.group('jobid')
        # If I am here, no valid line could be found.
        self.logger.error(f'in _parse_submit_output: unable to find the job id: {stdout}')
        raise SchedulerError(
            'Error during submission, could not retrieve the jobID from ketchup output; see log for more info.'
        )

    def _get_kill_command(self, jobid):
        """Return the command to kill the job with specified jobid."""

        kill_command = f'ketchup -t cancel {jobid}'

        _LOGGER.info(f'killing job {jobid}: {kill_command}')

        return kill_command

    def _parse_kill_output(self, retval, stdout, stderr):
        """Parse the output of the kill command.

        :return: True if everything seems ok, False otherwise.
        """
        if retval != 0:
            _LOGGER.error(f'Error in _parse_kill_output: retval={retval}; stdout={stdout}; stderr={stderr}')
            return False

        if stderr.strip():
            _LOGGER.warning(f'in _parse_kill_output there was some text in stderr: {stderr}')

        if stdout.strip():
            _LOGGER.warning(f'in _parse_kill_output there was some text in stdout: {stdout}')

        return True

    def parse_output(self, detailed_job_info, stdout, stderr):
        """Parse the output of the scheduler.

        :param detailed_job_info: dictionary with the output returned by the `Scheduler.get_detailed_job_info` command.
            This should contain the keys `retval`, `stdout` and `stderr` corresponding to the return value, stdout and
            stderr returned by the accounting command executed for a specific job id.
        :param stdout: string with the output written by the scheduler to stdout
        :param stderr: string with the output written by the scheduler to stderr
        :return: None or an instance of `aiida.engine.processes.exit_code.ExitCode`
        :raises TypeError or ValueError: if the passed arguments have incorrect type or value
        """
        raise NotImplementedError()
