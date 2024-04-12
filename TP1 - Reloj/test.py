import time
import datetime

class clock:
    def __init__ (self, hora = 0, minuto = 0, segundo = 0):
        self.hora = hora
        self.minuto = minuto
        self.segundo = segundo
    
    def onsec (self):

        n = int(100000000/13*9) # Ajusta este valor según la velocidad de tu sistema
        for i in range(n):
            pass

    def update (self):

        self.onsec() #pasa 1 segundo

        self.segundo += 1
        if self.segundo == 60:
            self.segundo = 0
            self.minuto += 1
            if self.minuto == 60:
                self.minuto = 0
                self.hora += 1
                if self.hora == 24:
                    self.hora = 0

    def printCurrentTime(self):

        print(f"The current time is:{self.hora:02d}:{self.minuto:02d}:{self.segundo:02d}")
        print("The real time is", datetime.datetime.now().time())
    def SetClock(self,tupla=None):
        '''
        Pasar como parametro una tupla con (hora, minuto, segundo)
        '''
        currentTime = tupla
        if currentTime == None: 
            currentTime = datetime.datetime.now()
            self.hora = currentTime.hour
            self.minuto = currentTime.minute
            self.segundo = currentTime.second
        
        else:
            self.hora = currentTime[0]
            self.minuto = currentTime[1]
            self.segundo = currentTime[2]

    def work(self, tupla=None):
        self.SetClock(tupla)
        print("Hora actualizada")
        while True:
            self.onsec()
            self.update()
            self.printCurrentTime()


class cronometro (clock):
    pass

class temporizador (clock):
    pass

reloj = clock()

reloj.SetClock()
reloj.work()