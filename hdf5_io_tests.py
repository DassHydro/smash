import smash
import numpy as np


setup, mesh = smash.load_dataset("cance")
model = smash.Model(setup, mesh)
model.run(inplace=True)

#save a single dictionary to hdf5
smash.tools.hdf5_handler.save_dict_to_hdf5("saved_dictionary.hdf5",mesh)

#generate the structure of the object: it is a dict of key:data to save: typeofstructure={light,medium,full}
keys_data=smash.io.hdf5_io.generate_smash_object_structure(model,typeofstructure="medium")
print(keys_data)
#add a new data to save:
keys_data["parameters"].append('ci')

#Save a single smash model
smash.save_smash_model_to_hdf5("./model_light.hdf5", model, content="light", replace=True)
smash.save_smash_model_to_hdf5("./model_medium.hdf5", model, content="medium", replace=True)
smash.save_smash_model_to_hdf5("./model_full.hdf5", model, content="full", replace=True)
smash.save_smash_model_to_hdf5("./model_user.hdf5", model, keys_data=keys_data, replace=True)

#adding subdata
sub_data={"sub_data1":"mydata"}
sub_data.update({"sub_data2":2.5})
sub_data.update({"sub_data3":{"sub_sub_data1":2.5,"sub_sub_data2":np.zeros(10)}})

smash.save_smash_model_to_hdf5("./model_sub_data.hdf5", model, content="medium",sub_data=sub_data, replace=True)


#view the hdf5 file
hdf5=smash.tools.hdf5_handler.open_hdf5("./model_user.hdf5")
hdf5.keys()
hdf5["mesh"].keys()
hdf5["parameters"].keys()
hdf5["output"].keys()
hdf5["output"].attrs.keys()
hdf5["output/fstates"].keys()
hdf5["setup"].attrs.keys()
hdf5.close()

#view the hdf5 file with sub_data
hdf5=smash.tools.hdf5_handler.open_hdf5("./model_sub_data.hdf5")
hdf5.keys()
hdf5.attrs.keys()
hdf5.close()


#save multi smash model at different place
smash.save_smash_model_to_hdf5("./multi_model.hdf5", model,location="model1",replace=True)
smash.save_smash_model_to_hdf5("./multi_model.hdf5", model,location="model2",replace=False)


hdf5=smash.tools.hdf5_handler.open_hdf5("./multi_model.hdf5")
hdf5.keys()
hdf5["model2"]["setup"].attrs.keys()
hdf5["model2"]["mesh"].keys()
hdf5["model2"]["output"].keys()
hdf5["model2"]["output"].attrs.keys()
hdf5.close()

#manually group different object in an hdf5
hdf5=smash.tools.hdf5_handler.open_hdf5("./model_subgroup.hdf5", replace=True)
hdf5=smash.tools.hdf5_handler.add_hdf5_sub_group(hdf5, subgroup="model1")
hdf5=smash.tools.hdf5_handler.add_hdf5_sub_group(hdf5, subgroup="model2")
keys_data=smash.io.hdf5_io.generate_smash_object_structure(model,typeofstructure="medium")
keys_data_2=smash.tools.object_handler.generate_object_structure(model)
smash.tools.hdf5_handler._dump_object_to_hdf5_from_iteratable(hdf5["model1"], model, keys_data)
smash.tools.hdf5_handler._dump_object_to_hdf5_from_iteratable(hdf5["model2"], model, keys_data_2)

hdf5=smash.tools.hdf5_handler.open_hdf5("./model_subgroup.hdf5", replace=False)
hdf5=smash.tools.hdf5_handler.add_hdf5_sub_group(hdf5, subgroup="model3")
keys_data=smash.io.hdf5_io.generate_smash_object_structure(model,typeofstructure="medium")
smash.tools.hdf5_handler._dump_object_to_hdf5_from_iteratable(hdf5["model3"], model, keys_data)

hdf5.keys()
hdf5["model1"].keys()
hdf5["model2"].keys()
hdf5["model3"].keys()
hdf5.close()


#read model object to a dictionnay
dictionary=smash.tools.object_handler.read_object_as_dict(model)
dictionary.keys()
dictionary["mesh"]["code"]

######### Reading HDF5

#load an hdf5 file to a dictionary
dictionary=smash.load_hdf5_file("./multi_model.hdf5")
dictionary["model1"].keys()
dictionary["model1"]["mesh"].keys()

#load a hdf5 file with any sub_data
dictionary=smash.load_hdf5_file("./model_sub_data.hdf5")
dictionary.keys()

#read only a part of an hdf5 file
hdf5=smash.tools.hdf5_handler.open_hdf5("./multi_model.hdf5")
dictionary=smash.tools.hdf5_handler.read_hdf5_as_dict(hdf5["model1"])
dictionary.keys()

#reload a full model object
model_reloaded=smash.load_hdf5_file("./model_medium.hdf5",as_model=True) #get error
model_reloaded=smash.load_hdf5_file("./model_full.hdf5",as_model=True)
model_reloaded
model_reloaded.run()

#TODO :

# compile documentation
# tests failed
# remove hdf5_io_test.py
# black *.py

