import torch
from torchvision import models

# Model Paths
MODEL_PATH = "fine_tuned_vit_imagenet100.pth"
OUTPUT_PATH = "fine_tuned_vit_imagenet100_scripted.pt"

# ImageNet-100 Class Labels (Full List of 100 Classes)
CLASS_NAMES = [
    'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead', 
    'electric_ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling',
    'goldfinch', 'house_finch', 'junco', 'indigo_bunting', 'robin', 
    'bulbul', 'jay', 'magpie', 'chickadee', 'water_ouzel', 'kite', 
    'bald_eagle', 'vulture', 'great_grey_owl', 'fire_salamander', 
    'smooth_newt', 'newt', 'spotted_salamander', 'axolotl', 'bullfrog', 
    'tree_frog', 'tailed_frog', 'loggerhead', 'leatherback_turtle', 
    'mud_turtle', 'terrapin', 'box_turtle', 'banded_gecko', 'common_iguana', 
    'American_chameleon', 'whiptail', 'agama', 'frilled_lizard', 'alligator_lizard', 
    'Gila_monster', 'green_lizard', 'African_chameleon', 'Komodo_dragon', 
    'triceratops', 'African_crocodile', 'American_alligator', 'thunder_snake', 
    'ringneck_snake', 'hognose_snake', 'green_snake', 'king_snake', 
    'garter_snake', 'water_snake', 'vine_snake', 'night_snake', 'boa_constrictor', 
    'rock_python', 'Indian_cobra', 'green_mamba', 'sea_snake', 'horned_viper', 
    'diamondback', 'sidewinder', 'trilobite', 'harvestman', 'scorpion', 
    'black_and_gold_garden_spider', 'barn_spider', 'garden_spider', 
    'black_widow', 'tarantula', 'wolf_spider', 'tick', 'centipede', 
    'black_grouse', 'ptarmigan', 'ruffed_grouse', 'prairie_chicken', 
    'peacock', 'quail', 'partridge', 'African_grey', 'macaw', 
    'sulphur-crested_cockatoo', 'lorikeet', 'coucal', 'bee_eater', 
    'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted_merganser', 
    'goose', 'black_swan', 'tusker', 'echidna', 'platypus', 'wallaby', 
    'koala', 'wombat'
]

# ✅ Recreate the ViT Model Architecture with Correct Number of Classes
print("Recreating ViT Model Architecture...")
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
model.heads.head = torch.nn.Linear(model.heads.head.in_features, len(CLASS_NAMES))

# ✅ Load State Dictionary
print("Loading the State Dictionary...")
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

# ✅ Adjust the Head Layer to Match State Dictionary
model.heads.head = torch.nn.Linear(model.heads.head.in_features, 100)
model.load_state_dict(state_dict)
model.eval()

# ✅ Convert to TorchScript
print("Converting to TorchScript...")
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, OUTPUT_PATH)

print(f"TorchScript Model saved at {OUTPUT_PATH}")
