from mappings import map_fine_to_model1, map_fine_to_model2, map_fine_to_model3, map_fine_to_model4
from utils import DNA_ALPHABET
import time

# check the label means it is coding seq
def is_coding_state(state):
    return state in {"C", "START", "EXON", "STOP", "C1", "C2", "C3"}

# check if label means it is splic seq
def is_splice_state(state):
    return state in {"DONOR", "ACCEPTOR"}

# see if it is classified right as coding or noncoding
def coding_metrics(pred, truth):
    tp = fp = tn = fn = 0
    # comparing prediciton and truth at each position
    for p, t in zip(pred, truth):
        p_c = is_coding_state(p)
        t_c = is_coding_state(t)

        if p_c and t_c:
            tp += 1
        elif p_c and not t_c:
            fp += 1
        elif not p_c and not t_c:
            tn += 1
        else:
            fn += 1

    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)

    return {
        "coding_accuracy": accuracy,
        "coding_sensitivity": sensitivity,
        "coding_specificity": specificity,
        "coding_precision": precision,
    }

# same as above but for introns
def intron_metrics(pred, truth):
    tp = fp = tn = fn = 0
    for p, t in zip(pred, truth):
        p_i = p in {"INTRON"}
        t_i = t in {"INTRON"}

        if p_i and t_i:
            tp += 1
        elif p_i and not t_i:
            fp += 1
        elif not p_i and not t_i:
            tn += 1
        else:
            fn += 1

    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)

    return {
        "intron_sensitivity": sensitivity,
        "intron_specificity": specificity,
        "intron_precision": precision,
    }

# detemrine how well splice sites are detected
def splice_site_metrics(pred, truth):
    # get indicies that the splic sites will ocurr at
    pred_splice = {i for i, lab in enumerate(pred) if is_splice_state(lab)}
    true_splice = {i for i, lab in enumerate(truth) if is_splice_state(lab)}

    overlap = pred_splice & true_splice

    recall = len(overlap) / max(1, len(true_splice))
    precision = len(overlap) / max(1, len(pred_splice))

    return {
        "splice_recall": recall,
        "splice_precision": precision,
    }

# separate evaluation of donor and accdeptor sites
def donor_acceptor_metrics(pred, truth):
    def positions(labels, target):
        return {i for i, lab in enumerate(labels) if lab == target}

    pred_donor = positions(pred, "DONOR")
    true_donor = positions(truth, "DONOR")
    pred_acceptor = positions(pred, "ACCEPTOR")
    true_acceptor = positions(truth, "ACCEPTOR")

    return {
        "donor_recall": len(pred_donor & true_donor) / max(1, len(true_donor)),
        "donor_precision": len(pred_donor & true_donor) / max(1, len(pred_donor)),
        "acceptor_recall": len(pred_acceptor & true_acceptor) / max(1, len(true_acceptor)),
        "acceptor_precision": len(pred_acceptor & true_acceptor) / max(1, len(pred_acceptor)),
    }

#idenitfiy contig coding regios with start and end regions
def detect_regions(labels):
    regions = []
    in_region = False
    start = 0

    for i, lab in enumerate(labels):
        if is_coding_state(lab) and not in_region:
            # start new coding region
            start = i
            in_region = True
        elif not is_coding_state(lab) and in_region:
            regions.append((start, i - 1))
            in_region = False

    if in_region:
        regions.append((start, len(labels) - 1))

    return regions

# comapred predited vs true regions but allowing tolerance of 3
def relaxed_boundary_matches(pred, truth, tolerance=3):
    pred_regions = detect_regions(pred)
    truth_regions = detect_regions(truth)

    matched_truth = set()
    matched_pred = set()

    #match rpedicted regions to true ones
    for pi, (p_start, p_end) in enumerate(pred_regions):
        for ti, (t_start, t_end) in enumerate(truth_regions):
            if ti in matched_truth:
                continue
            
            # see if bioundaries are close enoguh
            if abs(p_start - t_start) <= tolerance and abs(p_end - t_end) <= tolerance:
                matched_pred.add(pi)
                matched_truth.add(ti)
                break

    recall = len(matched_truth) / max(1, len(truth_regions))
    precision = len(matched_pred) / max(1, len(pred_regions))

    return {
        "boundary_recall": recall,
        "boundary_precision": precision,
    }


#evauate detection of start and stop
def start_stop_detection(pred, truth):
    def positions(labels, target):
        return {i for i, lab in enumerate(labels) if lab == target}

    pred_start = positions(pred, "START")
    true_start = positions(truth, "START")
    pred_stop = positions(pred, "STOP")
    true_stop = positions(truth, "STOP")

    return {
        "start_recall": len(pred_start & true_start) / max(1, len(true_start)),
        "start_precision": len(pred_start & true_start) / max(1, len(pred_start)),
        "stop_recall": len(pred_stop & true_stop) / max(1, len(true_stop)),
        "stop_precision": len(pred_stop & true_stop) / max(1, len(pred_stop)),
    }

#get list of all the average metrics
def average_metrics(metrics_list):
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}

# 3esztiamte the model complexiityh based on the number of state, transiton params, and emisions
def model_complexity(model):
    n_states = len(model.states)
    num_transitions = len(model.states) * len(model.states)
    num_emissions = len(model.states) * len(DNA_ALPHABET)
    return {
        "num_states": float(n_states),
        "num_transition_params": float(num_transitions),
        "num_emission_params": float(num_emissions),
    }

# mapfine-grained labels to label space by a model
def get_truth_for_model(example, model_name):
    if model_name.startswith("Model 1"):
        return map_fine_to_model1(example.fine_labels)
    if model_name.startswith("Model 2"):
        return map_fine_to_model2(example.fine_labels)
    if model_name.startswith("Model 3"):
        return map_fine_to_model3(example.fine_labels)
    if model_name.startswith("Model 4"):
        return map_fine_to_model4(example.fine_labels)

# evaluation loop for a single model
def _compute_metrics_for_model(model, dataset):
    all_coding_metrics = []
    all_boundary_metrics = []
    all_start_stop_metrics = []
    all_intron_metrics = []
    all_splice_metrics = []
    all_donor_acceptor_metrics = []

    for example in dataset:
        pred = model.viterbi(example.sequence)
        truth = get_truth_for_model(example, model.name)

        all_coding_metrics.append(coding_metrics(pred, truth))
        all_boundary_metrics.append(relaxed_boundary_matches(pred, truth, tolerance=3))
        all_start_stop_metrics.append(start_stop_detection(pred, truth))
        all_intron_metrics.append(intron_metrics(pred, truth))
        all_splice_metrics.append(splice_site_metrics(pred, truth))
        all_donor_acceptor_metrics.append(donor_acceptor_metrics(pred, truth))

    results = {}
    results.update(average_metrics(all_coding_metrics))
    results.update(average_metrics(all_boundary_metrics))
    results.update(average_metrics(all_start_stop_metrics))
    results.update(average_metrics(all_intron_metrics))
    results.update(average_metrics(all_splice_metrics))
    results.update(average_metrics(all_donor_acceptor_metrics))
    results.update(model_complexity(model))
    return results

# this evaluate the model and recordis runtime for how ling it takes for the model to run
def evaluate_model(model, dataset, timing_repeats=3, warmup=True):
    if warmup:
        _compute_metrics_for_model(model, dataset)

    timing_repeats = max(1, timing_repeats)
    total_runs = []

    for _ in range(timing_repeats):
        t0 = time.perf_counter()
        results = _compute_metrics_for_model(model, dataset)
        elapsed = time.perf_counter() - t0
        total_runs.append(elapsed)

    avg_total_runtime = sum(total_runs) / len(total_runs)
    results["total_runtime_seconds"] = avg_total_runtime
    results["timing_repeats"] = float(timing_repeats)
    results["num_examples_timed"] = float(len(dataset))

    return results

# run eval on all of the models
def benchmark_all_models(dataset, timing_repeats=3, warmup=True):
    from models import build_model1, build_model2, build_model3, build_model4

    models = [build_model1(), build_model2(), build_model3(), build_model4()]
    results = []

    for model in models:
        r = evaluate_model(
            model,
            dataset,
            timing_repeats=timing_repeats,
            warmup=warmup,
        )
        r["model_name"] = model.name
        results.append(r)

    return results