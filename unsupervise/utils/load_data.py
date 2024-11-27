import numpy as np

window_size=10

traindata_address = 'F:\\异常检测\\多维数据集\\多维数据集\\swat\\train_0.npz'
testdata_address = 'F:\\异常检测\\多维数据集\\多维数据集\\swat\\test_0.npz'

'''
traindata_n_o_address = 'F:\\异常检测\\多维数据集\\多维数据集\\swat\\train_n_o0.npz'
testdata_n_o_address = 'F:\\异常检测\\多维数据集\\多维数据集\\swat\\test_n_o0.npz'
'''

traindata = np.load(traindata_address, allow_pickle=True, encoding='bytes')
testdata = np.load(testdata_address, allow_pickle=True, encoding='bytes')

'''
traindata_n_o = np.load(traindata_n_o_address, allow_pickle=True, encoding='bytes')
testdata_n_o = np.load(testdata_n_o_address, allow_pickle=True, encoding='bytes')
'''


def get_train(data=traindata, is_recon=False, down_sample=False):

	train_data = data['window_data']#(T, w, d)
	#train_data = train_data[2160:]
	if is_recon == False:

		X_train = train_data[:, :window_size-1, :]#(T, w-1, d)
		Y_train = train_data[:, window_size-1, :]#(T,d)

	else:

		X_train, Y_train = train_data
	
	print('**************')
	print('load data done')
	return X_train, Y_train

'''
def get_train_n_o(data=traindata_n_o, is_recon=False):

	train_data = data['window_data']#(T/w, w, d)
	#train_data = train_data[2160:]
	if is_recon == False:

		X_train = train_data[:, :window_size-1, :]#(T/w, w-1, d)
		Y_train = train_data[:, window_size-1, :]#(T/w,d)

	else:

		X_train, Y_train = train_data

	print('**************')
	print('load data done')
	return X_train, Y_train
'''

def get_test(data=testdata, is_recon=False):

	test_data = data['window_data']#(t, w, d)
	test_label = data['window_label']#(t)
	#test_data = test_data[2160:]
	#test_label = test_label[2160:]
 
	if is_recon == False:

		X_test = test_data[:, :window_size-1, :]#(t, w-1, d)
		Y_test = test_data[:, window_size-1, :]#(t,d)

	else:

		X_test, Y_test = test_data
  
	print('**************')
	print('load data done')
	return X_test, Y_test, test_label#用model.predict(X_test)获得模型输出，与Y_test做差，根据阈值获得Y_label，与test_label比较

'''
def get_test_n_o(data=testdata_n_o, is_recon=False):

	test_data = data['window_data']#(t/w, w, d)
	test_label = data['window_label']#(t/w, w)
	#test_data = test_data[2160:]
	#test_label = test_label[2160:]
 
	if is_recon == False:

		X_test = test_data[:, :window_size-1, :]#(t/w, w-1, d)
		Y_test = test_data[:, window_size-1, :]#(t/w,d)

	else:

		X_test, Y_test = test_data
	print('**************')
	print('load data done')
	return X_test, Y_test, test_label#用model.predict(X_test)获得模型输出，与Y_test做差，根据阈值获得Y_label，与test_label比较
'''

if __name__ == '__main__':

    X_train, Y_train = get_train(traindata)
    X_test, Y_test, test_label = get_test(testdata)
    
    '''
	X_train, Y_train = get_train_n_o(traindata_n_o)
	X_test, Y_test, test_label = get_test_n_o(testdata_n_o)
    '''
    
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, test_label.shape)
    
