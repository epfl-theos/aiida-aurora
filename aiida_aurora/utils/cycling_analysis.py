import itertools
import json

from aiida.common.exceptions import AiidaException
from aiida.orm import CalcJobNode, QueryBuilder, RemoteData

from .parsers import get_data_from_raw, get_data_from_results


def analyze_cycling_results(data, consecutive_cycles, threshold, discharge):
    """Analyse cycling results.
    `data` should be a dictionary generated by `get_data_from_*`.
    """
    Qs = data['Qd'] if discharge else data['Qc']
    print(f"  capacities:          {Qs}")
    print(f"  relative capacities: {Qs / Qs[0]}")

    print(f"Completed {len(Qs)} cycles.")
    if len(Qs) >= consecutive_cycles + 1:
        below_thresh = Qs < threshold * Qs[0]
        below_groups = [sum(1 for _ in g) for k, g in itertools.groupby(below_thresh) if k]
        for g in below_groups:
            if g > consecutive_cycles:
                print(f'Below threshold for {g} cycles!')
    return data


def cycling_analysis(calcjob_node, retrieve_monitor_params=False, consecutive_cycles=2, threshold=0.8, discharge=True):
    """Perform the cycling analysis. You can provide either the cycler or the monitor calcjob.
    First, it will try to find and analyse any output of the cycler calcjob.
    If this does not succeed, the monitor calcjob outputs will be analysed (the results or the last snapshot).

      retrieve_monitor_params :  if True, try to load the monitor parameters from the inputs
    """

    monitor_calcjob = None
    if calcjob_node.process_type == 'aiida.calculations:aurora.cycler':
        calcjob = calcjob_node
        if calcjob.get_extra('monitored', False):
            # find last monitor, if existing
            qb = QueryBuilder()
            qb.append(RemoteData, filters={'uuid': calcjob.outputs.remote_folder.uuid}, tag='rf')
            qb.append(
                CalcJobNode,
                with_incoming='rf',
                edge_filters={'label': 'monitor_folder'},
                project=['*', 'id'],
                tag='mon'
            )
            qb.order_by({'mon': {'id': 'desc'}})
            monitor_calcjob = qb.first()[0] if qb.count() else None
    elif calcjob_node.process_type == 'aiida.calculations:calcmonitor.calcjob_monitor':
        monitor_calcjob = calcjob_node
        calcjob = monitor_calcjob.inputs.monitor_folder.get_incoming().get_node_by_label('remote_folder')
    else:
        raise TypeError('calcjob_node should be a BatteryCyclerExperiment or a CalcjobMonitor')

    if monitor_calcjob:
        print(f"Monitored CalcJob:   <{calcjob.id}> '{calcjob.label}'")
        print(f"Monitor CalcJob:     <{monitor_calcjob.id}> '{monitor_calcjob.label}'")
    else:
        print(f"CalcJob:             <{calcjob.id}> '{calcjob.label}'")

    sample = calcjob.inputs.battery_sample
    print(f"Sample:              {sample.label}")

    if monitor_calcjob:
        try:
            options = monitor_calcjob.inputs.monitor_protocols['monitor1'].get_attribute('options')
            threshold = options['threshold']
            discharge = (options['check_type'] == 'discharge_capacity')
            consecutive_cycles = options['consecutive_cycles']
        except AiidaException:
            # use default values
            pass
    print(f"Analysis options:")
    print(f"  check type:        ", "discharge capacity" if discharge else "charge capacity")
    print(f"  threshold:          {threshold}")
    print(f"  consecutive cycles: {consecutive_cycles}")

    def analyse_calcjob():
        output_labels = calcjob.get_outgoing().all_link_labels()
        if 'results' in output_labels:
            print('Analysing output results')
            res = calcjob.outputs.results
            return analyze_cycling_results(get_data_from_results(res), consecutive_cycles, threshold, discharge)
        elif 'raw_data' in output_labels and 'results.json' in calcjob.outputs.raw_data.list_object_names():
            print('Analysing output raw_data')
            jsdata = json.loads(calcjob.outputs.raw_data.get_object_content('results.json'))
            return analyze_cycling_results(get_data_from_raw(jsdata), consecutive_cycles, threshold, discharge)
        elif 'retrieved' in output_labels and 'results.json' in calcjob.outputs.retrieved.list_object_names():
            print('Analysing retrieved results.json file')
            jsdata = json.loads(calcjob.outputs.retrieved.get_object_content('results.json'))
            return analyze_cycling_results(get_data_from_raw(jsdata), consecutive_cycles, threshold, discharge)
        else:
            print('ERROR! CalcJob: no output found.')
            return None

    def analyse_monitor_calcjob():
        output_labels = monitor_calcjob.get_outgoing().all_link_labels()
        if 'redirected_outputs__results' in output_labels:
            print('Analysing redirected output results')
            res = monitor_calcjob.outputs.redirected_outputs.results
            return analyze_cycling_results(get_data_from_results(res), consecutive_cycles, threshold, discharge)
        elif 'redirected_outputs__raw_data' in output_labels and 'results.json' in monitor_calcjob.outputs.redirected_outputs.raw_data.list_object_names(
        ):
            print('Analysing redirected output raw_data')
            jsdata = json.loads(monitor_calcjob.outputs.redirected_outputs.get_object_content('results.json'))
            return analyze_cycling_results(get_data_from_raw(jsdata), consecutive_cycles, threshold, discharge)
        elif 'retrieved' in output_labels and 'results.json' in monitor_calcjob.outputs.retrieved.list_object_names():
            print('Analysing retrieved results.json file')
            jsdata = json.loads(monitor_calcjob.outputs.retrieved.get_object_content('results.json'))
            return analyze_cycling_results(get_data_from_raw(jsdata), consecutive_cycles, threshold, discharge)
        elif 'remote_folder' in output_labels:
            try:
                print('Analysing last snapshot.json file')
                with open(
                    f"{monitor_calcjob.outputs.remote_folder.get_attribute('remote_path')}/snapshot.json"
                ) as fileobj:
                    jsdata = json.load(fileobj)
                return analyze_cycling_results(get_data_from_raw(jsdata), consecutive_cycles, threshold, discharge)
            except FileNotFoundError:
                print('ERROR! Monitor CalcJob: no output found.')
                return None
        else:
            return None

    if not monitor_calcjob:
        # unmonitored job
        data = analyse_calcjob()
    else:
        # monitored job
        # NOTE this logic should be changed once we make sure that the monitor job/workchain always has the output
        # here I want to make sure that we do not read a snapshot, if the calcjob has an output
        # if it does not have an output, it means that it was killed by the monitor
        try:
            data = analyse_calcjob()
        except Exception as err:
            print(err)
            data = analyse_monitor_calcjob()
        else:
            if data is None:
                data = analyse_monitor_calcjob()
    return data
