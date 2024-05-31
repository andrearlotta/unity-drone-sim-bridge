#from tensorflow import keras
#import tensorflow as tf
from math import pi as pi

class NeuralClass:
    def __init__(self, model_path="/home/pantheon/lstm_sine_fitting/models/2024-03-05-17-51-46_LstmDenseDenseModel_lossmse_optadam_epochs1000_batch50_step1_input2xsequential_output4_dataset1000_noise1_0_unityTrue.keras",
                  loss='mse', opt='adam', weights=[1.0,10.0,180.0/pi,1.0]) -> None:
        self.net            = self.load()
        self.loss_dict      = { 'mse'       :   'mse'   }
        self.opt_dict       = { 'adam'      :   'adam'  }

        self.__setLoss(loss)
        self.__setOpt(opt)
        self.__setWeights(weights)
        self.__setModelPath(model_path)
        self.__loadModel()

    def __setLoss(self,loss):
        self.loss = self.loss_dict[loss]
    
    def __setOpt(self,opt):
        self.opt = self.opt_dict[opt]

    def __setWeights(self, weights):
        self.loss_weights = weights
    
    def __setModelPath(self, path):
        self.file_path = path

    def __getLoss(self):
        return self.loss
    
    def __getOpt(self):
        return self.opt

    def __getWeights(self):
        return self.loss_weights
    
    def __getModelPath(self):
        return self.file_path

    def __loadModel(self):
        self.model = keras.models.load_model(self.__getModelPath, compile=False)
        self.model.compile(optimizer=self.__getOpt, loss=self.__getLoss, loss_weights=self.__getWeights)
    
    def runPredict(self, input):
        return self.model.predict(input)[0]