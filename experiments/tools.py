from typing import Type
import mlflow.entities
import mlflow
import pm4py
from pm4py.objects.petri_net.exporter.variants.pnml import export_petri_as_string
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.powl import visualizer as powl_visualizer
from pm4py.objects.conversion.wf_net.variants.to_bpmn import apply as pn_to_bpmn
from pm4py.objects.bpmn.layout import layouter
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from powl.model_generator import ModelGenerator
import io
import sys


def store_powl(powl, folder: str):
    powl = powl.simplify()

    powl_svg = powl_visualizer.apply(powl, parameters={"format": "svg"})
    mlflow.log_text(powl_svg, f"{folder}/powl.svg")

    pn, im, fm = pm4py.convert_to_petri_net(powl)
    petri_net_svg = (
        pn_visualizer.apply(pn, im, fm, parameters={"format": "svg"})
        .pipe(format="svg")
        .decode("utf-8")
    )
    mlflow.log_text(petri_net_svg, f"{folder}/petri_net.svg")

    bpmn = pn_to_bpmn(pn, im, fm)
    bpmn = layouter.apply(bpmn)
    bpmn_svg = (
        bpmn_visualizer.apply(bpmn, parameters={"format": "svg"})
        .pipe(format="svg")
        .decode("utf-8")
    )
    mlflow.log_text(bpmn_svg, f"{folder}/bpmn.svg")


POWL_STORE = []


def get_last_powl_model():
    return POWL_STORE[-1]


GENERATED_MODELS = 0


def generate_powl_model(code: str) -> str:
    """Generates a POWL model using the given code, will return meaningful errors in case sth went wrong"""
    global GENERATED_MODELS

    # print(f"EXECUTING CODE:\n{code}")
    span = mlflow.get_current_active_span()
    n = GENERATED_MODELS
    GENERATED_MODELS += 1
    folder = f"{span.name}/{n}-{span.span_id}"

    span.set_attribute("POWL Model number:", n)
    span.set_attribute("reference", span.span_id)
    mlflow.log_text(code, f"{folder}/powl_model.py")

    output = io.StringIO()
    sys.stdout = output

    globals_dict = {"__builtins__": __builtins__, "ModelGenerator": ModelGenerator}

    locals_dict = {}

    try:
        exec(code, globals_dict, locals_dict)
        result = "Success"
    except Exception as e:
        result = str(e)
        if span:
            span.add_event(
                mlflow.entities.SpanEvent("exception", attributes={"message": result})
            )
        locals_dict = {}
    finally:
        sys.stdout = sys.__stdout__

    final_model = locals_dict.get("final_model", None)
    if final_model:
        store_powl(final_model, folder)
        POWL_STORE.append(final_model)
    return result
