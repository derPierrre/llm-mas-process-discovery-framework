import experiments.agents as agents
from agentFramework import link_agents
import mlflow
import mlflow.tracing
import mlflow.entities
from mlflow.types.chat import ChatMessage as MLFlowChatMessage
import subprocess
import concurrent.futures
import sys
import os


def all(experiment: str):
    models = ["vertex", "mistral", "deepseek"]
    processes = ["shop", "hotel", "reimbursement"]
    setups = ["monolithic", "duo", "manager", "team"]

    try:
        mlflow.create_experiment(experiment)
    except:
        pass

    # Run experiments for each model in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit work for each model
        future_to_model = {
            executor.submit(
                run_multiple_experiments_per_model, model, processes, setups, experiment
            ): model
            for model in models
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f"Experiments for model {model} generated an exception: {exc}")


def run_multiple_experiments_per_model(
    model: str, processes: list, setups: list, experiment: str
):
    current_file = os.path.abspath(__file__)
    current_module = os.path.splitext(os.path.basename(current_file))[0]

    print(f"Starting all experiments for model: {model}")

    for process in processes:
        for setup in setups:
            print(f"Running experiment on {model}: {process} - {setup}")

            python_code = f"""
import mlflow
from {current_module} import experiment

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('{experiment}')

with mlflow.start_run(run_name='{model}_{process}_{setup}'):
    experiment('{setup}', '{process}', '{model}')
"""
            cmd = [sys.executable, "-c", python_code]
            try:
                subprocess.run(cmd, check=True)
                print(f"Completed experiment on {model}: {process} - {setup}")
            except subprocess.CalledProcessError as e:
                print(f"Error running experiment on {model}: {process} - {setup}: {e}")

    return f"Completed all experiments for model: {model}"


def experiment(setup: str, process: str, model: str = "vertex"):
    mlflow.log_param("setup", setup)
    mlflow.log_param("process", process)
    mlflow.log_param("model", model)

    agents.MODEL = model
    history = []
    finished = False
    with mlflow.start_span(name="Simulation", span_type="CHAIN") as span:

        def log(sender, recipient, message):
            nonlocal span
            nonlocal history
            history.append(f"[{sender}] @ [{recipient}]\n{message}")
            print(f"{sender} @ {recipient}\n{message.strip()}\n--------------")
            mlflow.tracing.set_span_chat_messages(
                span,
                [
                    MLFlowChatMessage(
                        role="assistant", content=f"{sender} @ {recipient}\n\n{message}"
                    )
                ],
                True,
            )

        process_owners = agents.get_multiple_process_agents(process)

        if setup == "monolithic":
            agent = agents.monolithic()
            link_agents(
                agent,
                process_owners,
                interactionName="ask_process_owner",
                anonymizePrefix="Process Owner",
                callback=log,
            )
        elif setup == "duo":
            agent = agents.duo()
            link_agents(
                agent,
                process_owners,
                interactionName="ask_process_owner",
                anonymizePrefix="Process Owner",
                callback=log,
            )
        elif setup == "manager":
            agent = agents.manager()
            interviewer = next(
                (
                    a
                    for a in agent.get_all_subagents()
                    if a.name == "Knowledge Gatherer"
                ),
                agent,
            )
            link_agents(
                interviewer,
                process_owners,
                interactionName="ask_process_owner",
                anonymizePrefix="Process Owner",
                callback=log,
            )
        elif setup == "team":
            agent = agents.team(process_owners, log)
        else:
            raise ValueError(f"Invalid setup: {setup}")

        for a in agent.get_all_subagents():
            a.listen_to_all_deligations(log)

        try:
            agent.chat(
                "Start with discovering the business process.\n# Very important hint\nThe business process is simple, every process owner is only involved in a few activities, your job is to figure out how they work together.\nDont focus on deviations of the standard process or how the communication works.\nFigure out the big picture and the activities of the involved people and how they all work together."
            )
            finished = True
        except Exception as e:
            span.add_event(
                mlflow.entities.SpanEvent(
                    "exception",
                    attributes={"reason": "Error in simulation", "message": str(e)},
                )
            )

    history = [m.strip() for m in history]
    mlflow.log_text("\n------\n".join(history), "conversation.txt")

    if not finished:
        mlflow.end_run("FAILED")
