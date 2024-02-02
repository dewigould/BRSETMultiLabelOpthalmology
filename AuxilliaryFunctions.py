import numpy as np



def get_truths_and_preds(num_classes,model,test_data):

    test_preds = np.array([]).reshape((0,num_classes))
    test_true = np.array([]).reshape((0,num_classes))

    for i in range(test_data.__len__()):
        if CONFIG['use_weighted_loss_function']:
            x,y,z = test_data.__getitem__(i)
        else:
            x,y, = test_data.__getitem__(i)
        test_true = np.append(test_true,y,axis=0)
        test_preds = np.append(test_preds,model(x).numpy().reshape((1,num_classes)),axis=0)
    return test_true, test_preds

def split_dataset_by_camera_type(df_input, camera_type):
    if camera_type=='Canon':
        camera_name = "Canon CR"
    else:
        camera_name = "Nikon"
    labels_df = df[['image_id','camera','diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd', 'vascular_occlusion', 'hypertensive_retinopathy', 'drusens', 'hemorrhage', 'retinal_detachment', 'myopic_fundus', 'increased_cup_disc','other']]

    #Train/ Test / Val Split
    train_df_initial = labels_df[labels_df['camera']==camera_name]
    train_df_initial = train_df_initial.drop(columns=['camera'])
    train_inds, val_inds = train_test_split(np.array(list(range(train_df_initial.shape[0]))),test_size=0.2,random_state=2)
    train_df = train_df_initial.iloc[train_inds,:].reset_index(drop=True)
    val_df = train_df_initial.iloc[val_inds,:].reset_index(drop=True)
    test_df = labels_df[labels_df['camera']!=camera_name].reset_index(drop=True)
    test_df = test_df.drop(columns=['camera'])
    return train_df, val_df, test_df


def add_normals_to_test_set(arrays):
    result = []
    for inner_array in arrays:
        if np.all(inner_array == 0):
            result.append(np.append(inner_array, 1))
        else:
            result.append(np.append(inner_array, 0))
    return np.array(result)

def add_normals_to_pred_set(arrays, thresholds):
    result = []
    for inner_array in arrays:
        check=True
        for j in range(len(inner_array)):
            if inner_array[j] >= thresholds[j]:
                check=False
        if check:
            result.append(np.append(inner_array,1))
        else:
            result.append(np.append(inner_array,0))

    return np.array(result)


def check_columns_for_normality(row):
    for column in columns:
        if row[column] != 0:
            return 'abnormal'
    return 'normal'


def consolidate_cols_into_other(input_df,cols_to_remove):
    non_zero_condition = input_df[cols_to_remove].any(axis=1)
    input_df.loc[non_zero_condition, 'other'] = 1
    input_df = input_df.drop(columns=cols_to_remove)
    return input_df


def generate_ensemble_weight_tuples(n, length):
    tuples_list = []
    for _ in range(n):
        tuple_values = [random.random() for _ in range(length)]
        sum_values = sum(tuple_values)
        tuple_values = [value / sum_values for value in tuple_values]

        tuples_list.append(tuple(tuple_values))

    return tuples_list


def remove_random_subset_of_normal_entries(df,num):
    df['normality'] = df.apply(check_columns_for_normality, axis=1)

    normal_sample = df[df['normality']=='normal']
    normal_sample = normal_sample.sample(num)
    df = df.drop(normal_sample.index)
    df = df.drop(columns=['normality'])
    return(df)
