from azureml.core import Workspace, Model, Environment
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig

# Connect to the Azure workspace
workspace = Workspace(subscription_id="<your-subscription-id>",
                      resource_group="<your-resource-group>",
                      workspace_name="<your-workspace-name>")

# Retrieve the model tagged "deployed=true"
models = Model.list(workspace)
model_to_deploy = next((model for model in models if model.tags.get("deployed") == "true"), None)

if model_to_deploy:
    # Define the scoring script (score.py) and environment
    inference_config = InferenceConfig(
        entry_script="./score.py",   # Ensure score.py is in the same directory or specify the full path
        environment=Environment.get(workspace, name="pipeline-env")  # or create your own environment or stick to readme file
    )

    # Define deployment configuration
    deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

    # Deploy the model to ACI (can change to Azure Kubernetes Service if needed)
    service_name = "sentiment-analyser-endpoint"
    try:
        # Check if the service already exists
        service = Webservice(workspace, name=service_name)
        print("Service already exists, updating deployment...")

        # Update the service with the new model
        service.update(models=[model_to_deploy], inference_config=inference_config)
        service.wait_for_deployment(show_output=True)
    except Exception as e:
        print("Service does not exist, creating new deployment...")
        
        # Deploy the model to ACI
        service = Model.deploy(workspace=workspace,
                               name=service_name,
                               models=[model_to_deploy],
                               inference_config=inference_config,
                               deployment_config=deployment_config)
        service.wait_for_deployment(show_output=True)
    print(f"Deployment state: {service.state}")
    print(f"Scoring URI: {service.scoring_uri}")
else:
    print("No model tagged with 'deployed=true' found for deployment.")
