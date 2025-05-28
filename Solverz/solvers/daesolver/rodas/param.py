import numpy as np


class Rodas_param:

    def __init__(self,
                 scheme: str = 'rodas4'):
        match scheme:
            case 'rodas3':
                self.s = 4
                self.pord = 3
                self.alpha = np.zeros((self.s, self.s))
                self.beta = np.zeros((self.s, self.s))
                self.g = np.zeros((self.s, 1))
                self.gamma = 0.5
                self.alpha[1, 0] = 0
                self.alpha[2, 0:2] = [1, 0]
                self.alpha[3, 0:3] = [3/4, -1/4, 1/2]
                self.beta[1, 0] = 1
                self.beta[2, 0:2] = [3/4, -1/4]
                self.beta[3, 0:3] = [5/6, -1/6, -1/6]
                self.b = np.zeros((self.s,))
                self.b[0:3] = self.beta[-1, 0:3]
                self.b[3] = self.gamma
                self.bd = np.array([3/4, -1/4, 1/2, 0])
                self.c = np.zeros_like(self.b)
                self.d = np.zeros_like(self.b)
                self.e = np.zeros_like(self.b)
                self.gammatilde = self.beta - self.alpha
                self.a = np.sum(self.alpha, axis=1)
                self.g = np.sum(self.gammatilde, axis=1) + self.gamma
                self.gammatilde = self.gammatilde / self.gamma
                self.alpha = self.alpha.T
                self.gammatilde = self.gammatilde.T
            case 'rodas4':
                self.s = 6
                self.pord = 4
                self.alpha = np.zeros((self.s, self.s))
                self.beta = np.zeros((self.s, self.s))
                self.g = np.zeros((self.s, 1))
                self.gamma = 0.25
                self.alpha[1, 0] = 3.860000000000000e-01
                self.alpha[2, 0:2] = [1.460747075254185e-01, 6.392529247458190e-02]
                self.alpha[3, 0:3] = [-3.308115036677222e-01, 7.111510251682822e-01, 2.496604784994390e-01]
                self.alpha[4, 0:4] = [-4.552557186318003e+00, 1.710181363241323e+00, 4.014347332103149e+00,
                                      -1.719715090264703e-01]
                self.alpha[5, 0:5] = [2.428633765466977e+00, -3.827487337647808e-01, -1.855720330929572e+00,
                                      5.598352992273752e-01,
                                      2.499999999999995e-01]
                self.beta[1, 0] = 3.170000000000250e-02
                self.beta[2, 0:2] = [1.247220225724355e-02, 5.102779774275723e-02]
                self.beta[3, 0:3] = [1.196037669338736e+00, 1.774947364178279e-01, -1.029732405756564e+00]
                self.beta[4, 0:4] = [2.428633765466977e+00, -3.827487337647810e-01, -1.855720330929572e+00,
                                     5.598352992273752e-01]
                self.beta[5, 0:5] = [3.484442712860512e-01, 2.130136219118989e-01, -1.541025326623184e-01,
                                     4.713207793914960e-01,
                                     -1.286761399271284e-01]
                self.b = np.zeros((6,))
                self.b[0:5] = self.beta[5, 0:5]
                self.b[5] = self.gamma
                self.bd = np.zeros((6,))
                self.bd[0:4] = self.beta[4, 0:4]
                self.bd[4] = self.gamma
                self.c = np.array([-4.786970949443344e+00, -6.966969867338157e-01, 4.491962205414260e+00,
                                   1.247990161586704e+00, -2.562844308238056e-01, 0])
                self.d = np.array([1.274202171603216e+01, -1.894421984691950e+00, -1.113020959269748e+01,
                                   -1.365987420071593e+00, 1.648597281428871e+00, 0])
                self.e = np.zeros((6,))
                self.gammatilde = self.beta - self.alpha
                self.a = np.sum(self.alpha, axis=1)
                self.g = np.sum(self.gammatilde, axis=1) + self.gamma
                self.gammatilde = self.gammatilde / self.gamma
                self.alpha = self.alpha.T
                self.gammatilde = self.gammatilde.T
            case 'rodasp':
                self.s = 6
                self.pord = 4
                self.alpha = np.zeros((self.s, self.s))
                self.beta = np.zeros((self.s, self.s))
                self.gamma = 0.25
                self.alpha[1, 0] = 0.75
                self.alpha[2, 0:2] = [0.0861204008141522, 0.123879599185848]
                self.alpha[3, 0:3] = [0.774934535507324, 0.149265154950868, -0.294199690458192]
                self.alpha[4, 0:4] = [5.30874668264614, 1.33089214003727, -5.37413781165556, -0.265501011027850]
                self.alpha[5, 0:5] = [-1.76443764877448, -0.474756557206303, 2.36969184691580, 0.619502359064983,
                                      0.250000000000000]
                self.beta[1, 0] = 0.0
                self.beta[2, 0:2] = [-0.0493920000000000, -0.0141120000000000]
                self.beta[3, 0:3] = [-0.482049469387756, -0.100879555555556, 0.926729024943312]
                self.beta[4, 0:4] = [-1.76443764877448, -0.474756557206303, 2.36969184691580, 0.619502359064983]
                self.beta[5, 0:5] = [-0.0803683707891135, -0.0564906135924476, 0.488285630042799, 0.505716211481619,
                                     -0.107142857142857]
                self.b = np.zeros((6,))
                self.b[0:5] = self.beta[5, 0:5]
                self.b[5] = self.gamma
                self.bd = np.zeros((6,))
                self.bd[0:4] = self.beta[4, 0:4]
                self.bd[4] = self.gamma
                self.c = np.array([-40.98639964388325,
                                   -10.36668980524365,
                                   44.66751816647147,
                                   4.13001572709988,
                                   2.55555555555556,
                                   0])
                self.d = np.array([73.75018659483291,
                                   18.54063799119389,
                                   -81.66902074619779,
                                   -6.84402606205123,
                                   -3.77777777777778,
                                   0])
                self.e = np.zeros((6,))
                self.gammatilde = self.beta - self.alpha
                self.a = np.sum(self.alpha, axis=1)
                self.g = np.sum(self.gammatilde, axis=1) + self.gamma
                self.gammatilde = self.gammatilde / self.gamma
                self.alpha = self.alpha.T
                self.gammatilde = self.gammatilde.T
            case 'rodas5p':
                self.s = 8
                self.pord = 5
                self.a = np.zeros((self.s,))
                self.g = np.zeros((self.s,))
                self.alpha = np.array([
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.6358126895828704, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.31242290829798824, 0.09715693104176527, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.3140825753299277, 1.8583084874257945, -2.1954603902496506, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.42153145792835994, 0.25386966273009, -0.2365547905326239, -0.010005969169959593, 0.0, 0.0, 0.0,
                     0.0],
                    [1.712028062121536, 2.4456320333807953, -3.117254839827603, -0.04680538266310614,
                     0.006400126988377645, 0.0, 0.0, 0.0],
                    [-0.9993030215739269, -1.5559156221686088, 3.1251564324842267, 0.24141811637172583,
                     -0.023293468307707062, 0.21193756319429014, 0.0, 0.0],
                    [-0.003487250199264519, -0.1299669712056423, 1.525941760806273, 1.1496140949123888,
                     -0.7043357115882416, -1.0497034859198033, 0.21193756319429014, 0.0]
                ])
                self.beta = np.array([
                    [0.21193756319429014, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.21193756319429014, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-0.10952700614965587, -0.03129343032847311, 0.21193756319429014, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.701745865188331, -0.15675801606110462, 1.024650547472829, 0.21193756319429014, 0.0, 0.0, 0.0,
                     0.0],
                    [3.587261990937329, 1.6112735397639253, -2.378003402448709, -0.2778036907258995,
                     0.21193756319429014, 0.0, 0.0, 0.0],
                    [-0.9993030215739269, -1.5559156221686088, 3.1251564324842267, 0.24141811637172583,
                     -0.023293468307707062, 0.21193756319429014, 0.0, 0.0],
                    [-0.003487250199264519, -0.1299669712056423, 1.525941760806273, 1.1496140949123888,
                     -0.7043357115882416, -1.0497034859198033, 0.21193756319429014, 0.0],
                    [0.12236007991300712, 0.050238881884191906, 1.3238392208663585, 1.2643883758622305,
                     -0.7904031855871826, -0.9680932754194287, -0.214267660713467, 0.21193756319429014]
                ])
                self.c = np.array([-0.8232744916805133, 0.3181483349120214, 0.16922330104086836, -0.049879453396320994,
                                   0.19831791977261218, 0.31488148287699225, -0.16387506167704194,
                                   0.036457968151382296])
                self.d = np.array([-0.6726085201965635, -1.3128972079520966, 9.467244336394248, 12.924520918142036,
                                   -9.002714541842755, -11.404611057341922, -1.4210850083209667,
                                   1.4221510811179898])
                self.e = np.array([1.4025185206933914, 0.9860299407499886, -11.006871867857507, -14.112585514422294,
                                   9.574969612795117, 12.076626078349426, 2.114222828697341,
                                   -1.0349095990054304])
                self.gamma = self.beta[0, 0]
                self.b = np.append(self.beta[7, :-1], [self.gamma])
                self.bd = np.append(self.beta[6, :-2], [self.gamma, 0])
                self.gammatilde = self.beta - self.gamma * np.eye(self.s) - self.alpha
                for i in range(self.s):
                    self.a[i] = np.sum(self.alpha[i, :])
                    self.g[i] = np.sum(self.gammatilde[i, :]) + self.gamma
                self.gammatilde /= self.gamma
                self.alpha = self.alpha.T
                self.gammatilde = self.gammatilde.T
            case 'rodas3d':
                self.s = 4
                self.pord = 3
                self.alpha = np.zeros((self.s, self.s))
                self.beta = np.zeros((self.s, self.s))
                self.g = np.zeros((self.s, 1))
                self.gamma = 0.57281606
                self.alpha[1, 0] = 1.2451051999132263
                self.alpha[2, 0:2] = [1, 0]
                self.alpha[3, 0:3] = [0.32630307266483527, 0.10088086733516474, 0.57281606]
                self.beta[1, 0] = -3.1474142698552949
                self.beta[2, 0:2] = [0.32630307266483527, 0.10088086733516474]
                self.beta[3, 0:3] = [0.69775271462407906, 0.056490613592447572, -0.32705938821652658]
                self.b = np.zeros((self.s,))
                self.b[0:3] = self.beta[3, 0:3]
                self.b[self.s - 1] = self.gamma
                self.bd = np.zeros((self.s,))
                self.bd[0:2] = self.beta[2, 0:2]
                self.bd[self.s - 2] = self.gamma
                # self.c = np.array([-4.786970949443344e+00, -6.966969867338157e-01, 4.491962205414260e+00,
                #                    1.247990161586704e+00, -2.562844308238056e-01, 0])
                # self.d = np.array([1.274202171603216e+01, -1.894421984691950e+00, -1.113020959269748e+01,
                #                    -1.365987420071593e+00, 1.648597281428871e+00, 0])
                # self.e = np.zeros((6,))
                self.gammatilde = self.beta - self.alpha
                self.a = np.sum(self.alpha, axis=1)
                self.g = np.sum(self.gammatilde, axis=1) + self.gamma
                self.gammatilde = self.gammatilde / self.gamma
                self.alpha = self.alpha.T
                self.gammatilde = self.gammatilde.T
            case _:
                raise ValueError("Not implemented")
