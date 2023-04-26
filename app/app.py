import streamlit as st
import string
import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from sentence_transformers import SentenceTransformer, util
import templates as tp
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from tensorflow.keras import utils as np_utils
from keras.utils.np_utils import to_categorical

import transformers
from transformers import AutoTokenizer,TFDistilBertModel, DistilBertConfig
from transformers import TFAutoModel
 
# load model
from keras.models import load_model


loaded_model = tf.keras.models.load_model('modelv1.h5',custom_objects={'TFDistilBertModel':TFDistilBertModel}) 

target_classes = ['a/c_and_climatronic_control_panels', 'abs_dsc_asr',
       'ac_compressors', 'active_suspension_modules',
       'active_suspension_self_leveling_shock_absorbers_shock_dampers',
       'aerials', 'air_conditioning_radiators', 'air_filter_boxes',
       'air_horns_klaxons_buzzers', 'air_hoses_induction_pipes',
       'air_intake_smooth_rubber_hoses', 'alloy_rims', 'alternators',
       'amplifiers', 'armrests', 'automatic_transmissions',
       'blinds_and_mechanisms', 'blower_motor_resistors', 'bonnet_locks',
       'boot_lid_key_locks', 'boot_lid_locks', 'brake_calipers',
       'brake_discs', 'brake_master_cylinders',
       'bumper_shock_absorbers_support_frames_and_holders',
       'buttons_panel_(seats_windows_mirrors_etc.)', 'carburetors',
       'cassette_and_cd_players', 'cd_changers',
       'central_vacuum_and_door_lock_actuators', 'coil_springs',
       'commutators',
       'compressors_for_air_suspension_hydraulic_suspension_pumps',
       'control_buttons_switch_hazard', 'conveying_pumps',
       'coolant_reservoirs', 'cooling_fans', 'crankcases', 'cup_holders',
       'cylinder_heads', 'dampers_for_bonnets_lids_doors_and_glove_boxes',
       'decorative_masks_and_grilles', 'delco_distributors',
       'diesel_fuel_injectors',
       'differential_and_transfer_case_actuators', 'differentials',
       'displays_and_gauges', 'door_locks', 'door_panels', 'doors',
       'drive_shafts', 'driveshafts_propshafts', 'dvd_tv_receivers',
       'egr_and_vacuum_valves', 'electric_steering_rack_motors',
       'emergency_lights_buttons', 'engine_control_units',
       'engine_mounts', 'exterior_handles', 'external_boot_lid_handles',
       'fan_clutches', 'fan_shrouds', 'fenders', 'flywheels',
       'for_air_conditioner', 'for_antifreeze', 'for_oil_and_hydraulics',
       'front_axles', 'fuel_burning_heaters', 'fuel_hoses_and_pipes',
       'fuel_injection_pumps', 'fuel_injection_systems',
       'fuel_level_sensors', 'fuel_pump_chambers_and_flangers',
       'fuel_pumps', 'fuel_rails', 'fuse_panels',
       'gasoline_fuel_injectors',
       'gear_shift_lever_knobs_parking_brake_handles_and_leather_covers',
       'gearbox_cables', 'gearbox_control_modules', 'glove_compartments',
       'glow_plugs_coolant_and_fuel_pre_heaters', 'gps_navigations',
       'handbrake_and_gear_shift_levers', 'hands_free',
       'headlight_wipers_and_mechanisms',
       'heater_motor_flap_control_and_heater_valves', 'heater_vents',
       'heating_blowers', 'heating_control_panels', 'heating_radiators',
       'hoses_and_pipes_for_turbo', 'idle_speed_motors_and_swirl_flaps',
       'ignition_coils',
       'ignition_keys_security_cylinder_locks_and_ignition_switch_connectors',
       'inner_fenders_mud_flaps', 'instrument_clusters', 'intercoolers',
       'interior_courtesy_lights', 'interior_dash_trim_panels',
       'internal_handles', 'leaf_springs', 'light_switches',
       'loudspeakers', 'manifolds_intake_exhaust',
       'manuals_transmissions', 'mass_air_flow_meter_sensors', 'mirrors',
       'modules', 'nozzles_for_windscreens_headlights', 'oil_coolers',
       'oil_coolers_and_heat_exchangers', 'oil_pumps',
       'park_assist_sensors', 'pistons_cylinder_sleeves',
       'plastic_panels_and_elements', 'power_motor_window_regulators',
       'power_steering_pumps', 'pulleys_and_tensioners', 'rear_axles',
       'relays', 'rods', 'roof_racks_luggage_rack_roof_rails',
       'seat_belts', 'sensors', 'servo', 'shafts', 'side_skirts',
       'side_windows', 'skid_plates', 'spare_wheel_holders_tool_box_kits',
       'spare_wheels_with_tires', 'spoilers', 'starters', 'steel_rims',
       'steering_boxes', 'steering_shafts_steering_wheel_joints',
       'steering_wheel_ribbon_cables', 'steering_wheels', 'subwoofers',
       'summer_tires', 'sun_visors', 'sunroofs',
       'suspension_spheres_valves_hydraulic_suspension_distributors',
       'thermostats', 'throttle_potentiometers', 'throttles',
       'tool_kits_and_lifting_jacks', 'torque_converters', 'towing_hooks',
       'transfer_cases', 'trims_and_caps__external',
       'trunk_interior_covers', 'turbos_and_actuators', 'vacuum_pumps',
       'variators_oil_valves',
       'water_connections_and_thermostat_housings',
       'water_pump_heater_coolant_motors', 'water_radiators',
       'windshield_washer_pumps', 'windshield_wiper_mechanisms',
       'windshield_wipers_reservoirs', 'winter_tires',
       'wipers_and_lights_levers', 'wirings']



def getData(search_string):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    token =tokenizer(search_string,return_tensors='tf', truncation=True, padding='max_length',max_length=100, add_special_tokens=True)
    input_ids = token['input_ids']
    attention_mask = token['attention_mask']
    output =loaded_model.predict([input_ids,attention_mask])
    outputIndex = tf.math.argmax(output, axis=-1)[0]
    data = target_classes[outputIndex]
    print("Data from ES: ")
    print(data)
    return data




# nltk library Stop words list
stop_words = {'hasnt', 'for', 'ma', 'up', 'should', 'which', 'now', 'her', 'so', 'these', 'don', 'll', 'youd', 'against', 'doing', 'my', 'mightnt', 'him', 'but', 'is', 'dont', 'shouldve', 'arent', 'then', 'during', 't', 'above', 'once', 'shouldn', 'we', 'themselves', 're', 'was', 'needn', 'herself', 'has', 'be', 'as', 'from', 'until', 'between', 'his', 'hadn', 'mustn', 'under', 'too', 'through', 'mustnt', 'can', 'ours', 'theirs', 'me', 'you', 'shouldnt', 'she', 'over', 'or', 'isn', 'in', 'your', 'haven', 'ourselves', 'again', 'further', 'when', 'no', 'o', 'he', 'what', 'himself', 'all', 'after', 'will', 'been', 'have', 'not', 'being', 'other', 'having', 'few', 'both', 'than', 'that', 'it', 'some', 'about', 'their', 'whom', 'its', 'are', 'had', 'out', 'into', 'where', 've', 'our', 'the', 'youve', 'them', 'nor', 'just', 'while', 'am', 'down', 'd', 'm', 'of', 'doesnt', 'why', 'hers', 'shant', 'wasn', 'havent', 'hadnt', 'aren', 'wouldnt', 'who', 'by', 'here', 'shan', 'didn', 'such', 'own', 'below', 'neednt', 'same', 'if', 'off', 'myself', 'a', 'each', 'this', 'thatll', 'how', 'youre', 'does', 'yourselves', 'do', 'very', 'isnt', 'any', 'wont', 'werent', 'those', 'because', 'yourself', 'y', 'won', 'did', 'at', 'couldnt', 'more', 'its', 'there', 'with', 'on', 'itself', 'only', 'an', 'before', 'mightn', 'yours', 'ain', 'they', 'wasnt', 'were', 'doesn', 'shes', 'weren', 'most', 'didnt', 'hasn', 'youll', 'wouldn', 'to', 's', 'i', 'couldn', 'and'}


# make all text lowercase
def text_lowercase(text):
    return text.lower()


# remove stopwords
def remove_stopwords(text):
    words = text.split()
    text = [i for i in words if not i in stop_words]
    text = ' '.join(text)
    return text


# remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


# preprocess text string
def preprocessing(text):
    text = text_lowercase(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    return text


    
def main():
    st.title('Product Categorization')
    search = st.text_area('Enter Product description:')
    print("search string: "+search)
    data = [{"product_description" : ''}]
    if search:
        data = getData(search)
        results = data
        print(results)
    # render popular tags as filters
        st.write(tp.search_result(1, results), unsafe_allow_html=True)
        

if __name__ == '__main__':
    main()