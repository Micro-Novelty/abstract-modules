import asyncio
import time
import AbstractIntegratedModule
from AbstractIntegratedModule import (
    IntegratedPipeline,
    PipelinePredictionManager,
    CohesiveAgentDeployment
)
import numpy as np
from sklearn.datasets import make_classification, make_moons, make_circles, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split


print(f'FILE: {AbstractIntegratedModule.__file__}')
memory_name = 'agent_memory'
main_model = IntegratedPipeline(memory_name=memory_name, use_async=True, agent_port=8080,ssl_cert_file=None, ssl_key_file=None) # provide cert_file path or key_file path (optional)
main_prediction = PipelinePredictionManager(main_model, label_csv='ManualsTraining.txt', target_title='window_title', label='label')
# example_manual_training is a .txt file that contain csv format like above example.

example_rules = [
                     # === WORK / PRODUCTIVITY ===
                     (r'code|programming|develop|debug|compile|script', 'focused_work'),
                     (r'vscode|visual_studio|ide|terminal|shell', 'focused_work'),
                     (r'notion|evernote|onenote|notes|todo|task', 'productive'),
                     (r'slack|teams|discord|zoom|meeting|call', 'communication'),
                     (r'email|gmail|outlook|inbox|mail', 'communication'),
                     
                     # === ENTERTAINMENT ===
                     (r'youtube|netflix|twitch|stream|video', 'entertainment'),
                     (r'music|spotify|soundcloud|audio|player', 'entertainment'),
                     (r'game|gaming|steam|epic|play', 'gaming'),
                     (r'facebook|instagram|tiktok|social|post', 'social_media'),
                     
                     # === BROWSING ===
                     (r'chrome|firefox|edge|safari|browser', 'browsing'),
                     (r'google|search|wiki|wiki|article', 'information'),
                     (r'stackoverflow|github|docs|documentation', 'research'),

                     # more rules
                 ]
# activate explainability capability to explain uncertainty:
main_model.show_explainability_details = False
main_model.distribution.predict_manager = main_prediction
main_model.autonomous = True

# deprecate transformer use.
main_model.use_transformer = False

# test samples with more sophisticated rules and more complex titles for prediction
# (title, intent)
test_titles = [
 ("Opening Thesis.docx", "slight_work"),
 ("Watching YouTube and Google Chrome", "distracted"),
 ("Watching Slack", "communication"),
 ("Programming in Visual Studio Code", "focused_work"),
 ("Watching netflix.com - Chrome", "break"),
# more titles 
 ]  

def test_sets(type=None):
    dt = 0
    if type == "moons" or dt > 5:
    	X, y_raw = make_moons(
    	n_samples = 1000, 
    	noise=0.75,  
    	random_state=99)
    else:
    	dt += 1    
    	if dt <= 5:
    		print("type sample 1")
    		X, y_raw = make_classification(
    		n_samples=1000,
    		n_features=3,
    		n_classes=3,
    		n_informative=3,
    		n_redundant=0,
    		class_sep=1.5,
    		random_state=259)
    	else:
    		generalization_test()
    	
    return X, y_raw

X, y = test_sets()
            
titles, _, label_map = main_prediction.load_labels_from_csv('ManualsTraining.txt', 'window_title', 'label')
# small training with simple titles
main_model.train(titles, y)

y = np.asarray(y)
print(X.shape, y.shape)
   
results, chosen_label, confidence = main_prediction.advanced_prediction_method(test_titles, label_map, example_rules,
                             X=X, y=y,
                             show_proba=False, top_k=3, 
                             use_transformer=True,
                             return_attention=False,
                             save_results=True,
                             batch_size=2)


dataset, _ = main_model.data_preparation(titles, label_map)
sequence_inputs = main_model.sequence_encoding(dataset)
X_raw_generation, y, n_classes, input_dim = main_model.mlp_training_features(example_rules, dataset)

main_model.initialize_fitting(dataset)
X_raw_features = main_model.tfidf.transform(X_raw_generation).toarray()
transformer_features = main_model.transformer_pooled_features(sequence_inputs)
X_features = np.concatenate([X_raw_features, transformer_features], axis=-1)

peer_probability_calibration = main_model.predict_proba(sequence_inputs, X_features, type='Hybrid', embedded=True) # peer-to-peer calibration is inside this function

from AbstractIntegratedModule import PipelineAsyncManager
from AbstractIntegratedModule import SecurityConfig
from AbstractIntegratedModule import SecurityLevel

print(" = TESTING ASYNCHRONOUS PREDICTION MANAGER = ")
# Set discovery secret (in production, use environment variable)
secret_key = 'my-ultra-safe-secret-key-for-authentication' # you can customize this key


security_config = SecurityConfig(
      max_text_length=10000, # can be extended
      max_queue_size=100, # can be extended
      rate_limit_requests=60,  # 60 per minute
      require_api_key=True, #
      max_pending_tasks=50,
      request_timeout=60.0,

      # Start with no IP restrictions, you can add allowed IPs for asynchronous prediction externally, boothstrap_auth for better security
      allowed_ips=[],
      blocklisted_ips=[],
      require_bootstrap_auth = False # true for better security (Not recommended, cause less flexibility)
  )

async_manager = PipelineAsyncManager(main_model, 
        main_prediction, # your previous initialized PipelinePredictionManager
        config=security_config, 
        state_file=None, # state file is used to load known security logs ex: ip used, ip blacklisted, etc.
        security_level=SecurityLevel.PRODUCTION, # production level security initiated
        api_key=secret_key, #set secret key you initialized        
        max_workers=4, # workers to initiate prediction, more workers, more capabilities to process prediction requests.
        task_timeout=30, 
        max_retries=3 ) # retries after failure during prediction

async_manager.start(method='Transformer_included', bootstrap_token=None) # boothstrap token is optional for better security

texts = {'test_titles': test_titles, 'label_map': label_map, 'rules': example_rules, 'X': None, 'y': None, 'use_transformer': True}
regular_predict = async_manager.predict(
   texts=texts,
   timeout=60,
   retries=None,
   api_key=secret_key) # advanced prediction method for asynchronous prediction.

# with retries: async_manager.predict(texts, timeout=60, retries=5, api_key=secret_key) # 5 times retry if failed

print('[==] Initiating advanced batch prediction')
predicted_output = async_manager.advanced_batch_prediction(test_titles, label_map, example_rules, api_key=secret_key, client_ip=None)
# for better and faster advanced prediction, consider using advanced batch prediction like in the above example


# ... more features you can add

# ================ SECOND TEST ================
SECRET_KEY = "test-secret-key"

LABEL_FILE = "ManualsTraining.txt"
PEER_FILE = "peer_config.json"

memory_name = "arm64_test"

pipeline = IntegratedPipeline(memory_name=memory_name, use_async=True, agent_port=8080)

predict_manager = PipelinePredictionManager(
    pipeline,
    label_csv=LABEL_FILE,
    target_title="window_title",
    label="label"
)

titles, y, label_map = predict_manager.load_labels_from_csv(
    LABEL_FILE,
    "window_title",
    "label"
)

pipeline.train(titles, y)

test_titles = [
    ("Programming in VSCode", "focused_work"),
    ("Watching YouTube", "entertainment")
]

print('FILE:', AbstractIntegratedModule.__file__)


async def main(main_pipeline):
    sec_pipeline = IntegratedPipeline(memory_name=memory_name, use_async=True, agent_port=8010)

    rules = [
        # === WORK / PRODUCTIVITY ===
        (r'code|programming|develop|debug|compile|script', 'focused_work'),
        (r'vscode|visual_studio|ide|terminal|shell', 'focused_work'),
        (r'notion|evernote|onenote|notes|todo|task', 'productive'),
        (r'slack|teams|discord|zoom|meeting|call', 'communication'),
        (r'email|gmail|outlook|inbox|mail', 'communication'),
        
        # === ENTERTAINMENT ===
        (r'youtube|netflix|twitch|stream|video', 'entertainment'),
        (r'music|spotify|soundcloud|audio|player', 'entertainment'),
        (r'game|gaming|steam|epic|play', 'gaming'),
        (r'facebook|instagram|tiktok|social|post', 'social_media'),
        
        # === BROWSING ===
        (r'chrome|firefox|edge|safari|browser', 'browsing'),
        (r'google|search|wiki|wiki|article', 'information'),
        (r'stackoverflow|github|docs|documentation', 'research'),
        
        # === FILE MANAGEMENT ===
        (r'download|folder|file|document|pdf', 'file_work'),
        (r'dropbox|onedrive|google_drive|cloud', 'cloud_storage'),
        (r'zip|rar|extract|compress|archive', 'file_management'),
        
        # === SYSTEM / DEV ===
        (r'terminal|cmd|powershell|bash|shell', 'system_work'),
        (r'docker|kubernetes|container|deploy', 'devops'),
        (r'git|commit|push|pull|branch|merge', 'version_control'),
        (r'test|unit|debug|error|exception', 'testing'),
        
        # === DATA / ANALYSIS ===
        (r'excel|spreadsheet|sheet|csv|table', 'data_work'),
        (r'python|r|sql|query|database', 'data_analysis'),
        (r'chart|graph|visualization|dashboard|plot', 'visualization'),
        
        # === COMMUNICATION ===
        (r'whatsapp|telegram|signal|messenger', 'messaging'),
        (r'zoom|meet|webex|video_call', 'video_call'),
        (r'calendar|schedule|event|meeting|appointment', 'scheduling'),
        
        # === CREATIVE ===
        (r'photoshop|illustrator|figma|design|canvas', 'creative'),
        (r'premiere|final_cut|video_edit|render', 'video_editing'),
        (r'blender|3d|model|render|animation', '3d_work'),
        
        # === LEARNING ===
        (r'coursera|udemy|edx|course|learn', 'learning'),
        (r'book|ebook|reader|pdf|document', 'reading'),
        (r'podcast|audiobook|listen|lecture', 'audio_learning'),
        
        # === UTILITY ===
        (r'calculator|converter|tool|utility', 'utility'),
        (r'weather|clock|timer|alarm|reminder', 'utility'),
        (r'translate|language|dictionary|translate', 'utility'),
        
        # === RARITY PATTERNS ===
        (r'common|not_common|twitch|debian|watch', 'very abundant'),
        (r'bit-common|pycharm|unix|code|programming|python|java', 'bit-abundant'),
        (r'medium|discord|teams|zoom|linux_mint|message', 'abundant'),
        (r'rare|pdf|word|macOS|ubuntu|document', 'not abundant'),
        (r'ultra|firefox|edge|browser|unix|web', 'medium rare'),
        (r'ultra_rare|music|linux|Home_linux_router', 'bit-rare'),
        (r'medium-rare|steam|red_hat_enterprise_linux|play|windows', 'very rare'),
        (r'rarer|oracle|system|config|server_linux_router', 'absolute rare'),
    ]

    agent1 = CohesiveAgentDeployment(
        pipeline=main_pipeline,
        memory_name="agent1",
        filename=LABEL_FILE,
        target_title="window_title",
        label_name="label",
        enable_peers=True,
        peer_discovery_port=5555,
        secret_key=SECRET_KEY,
        shared_auth_token=SECRET_KEY,
        predict_manager=predict_manager,
        peer_config=PEER_FILE,
    )

    agent2 = CohesiveAgentDeployment(
        pipeline=sec_pipeline,
        memory_name="agent2",
        filename=LABEL_FILE,
        target_title="window_title",
        label_name="label",
        enable_peers=True,
        peer_discovery_port=5556,
        secret_key=SECRET_KEY,
        shared_auth_token=SECRET_KEY,
        predict_manager=predict_manager,
        peer_config=PEER_FILE,
    )

    await agent1.start()
    print("Agent 1 started on port 5555, Time:", time.time())
    await agent2.start()
    print("Agent 2 started on port 5556, Time:", time.time())

    await asyncio.sleep(3)

    api_key = agent1.get_api_key()

    texts = {
        "test_titles": test_titles,
        "label_map": label_map,
        "rules": rules,
        "X":None,
        "y":None,
        "use_transformer": False,
        "agent_id": "arm64-demo"
    }

    # agent1.pipeline.use_transformer = False
    # agent2.pipeline.use_transformer = False

    result = await agent1.multi_modal_peer_ensemble_prediction(
        texts=texts,
        api_key=api_key,
        method="advanced",
        disable_sync=True
    )

    result2 = await agent2.multi_modal_peer_ensemble_prediction(
        texts=texts,
        api_key=api_key,
        method="advanced",
        disable_sync=True
    )    

    print("\n=== RESULT ===")
    print(f"\n📊 Ensemble Result for Agent 1:")
    print(f"   Prediction: {result.get('prediction', 'N/A')}")
    print(f"   Confidence: {result.get('confidence', 0):.2%}")

    print('[=] For agent 2: ')
    print(f"   Second Prediction: {result2.get('prediction', 'N/A')}")
    print(f"   Second Confidence: {result2.get('confidence', 0):.2%}")

    agent1._peer_agent.stop_server()
    agent2._peer_agent.stop_server()

    await agent1.shutdown()
    await agent2.shutdown()

asyncio.run(main(pipeline))
