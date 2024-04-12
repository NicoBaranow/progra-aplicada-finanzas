import datetime
import time
import threading

class Clock:
    def __init__(self, hour=0, minute=0, second=0, is_24hour=True):
        self.hour = hour
        self.minute = minute
        self.second = second
        self.is_24hour = is_24hour
        self.running = True

    def onesec(self):
        # Incrementa un segundo
        time.sleep(1)

    def update(self):
        # Actualiza el tiempo
        self.second += 1
        if self.second == 60:
            self.second = 0
            self.minute += 1
            if self.minute == 60:
                self.minute = 0
                self.hour += 1
                if self.hour == 24 and self.is_24hour:
                    self.hour = 0
                elif self.hour == 12 and not self.is_24hour:
                    self.hour = 0

    def printCurrentTime(self):
        # Imprime la hora actual
        if self.is_24hour:
            print("The current time is: {:02d}:{:02d}:{:02d}".format(self.hour, self.minute, self.second))
        else:
            am_pm = "AM" if self.hour < 12 else "PM"
            hour = self.hour if self.hour <= 12 else self.hour - 12
            print("The current time is: {:02d}:{:02d}:{:02d} {}".format(hour, self.minute, self.second, am_pm))

    def SetClock(self, tupla=None):
        # Inicializa el reloj
        if tupla is None:
            current_time = datetime.datetime.now()
            self.hour = current_time.hour
            self.minute = current_time.minute
            self.second = current_time.second
        else:
            self.hour, self.minute, self.second = tupla

    def work(self, tupla=None):
        # Inicia el reloj y permite interactuar con él
        self.SetClock(tupla)
        key_thread = threading.Thread(target=self.handle_keys)
        key_thread.start()
        while self.running:
            self.onesec()
            self.update()

    def handle_keys(self):
        while self.running:
            key = input("Press 'p' to print current time or 'q' to quit: ")
            if key == 'p':
                self.printCurrentTime()
            elif key == 'q':
                self.running = False
                break

class Cronometro(Clock):
    def __init__(self, hour=0, minute=0, second=0, is_24hour=True):
        super().__init__(hour, minute, second, is_24hour)
        self.times = []  # Lista para almacenar los tiempos de cada vuelta

    def lap(self):
        # Registra el tiempo de vuelta actual y lo imprime
        lap_time = (self.hour, self.minute, self.second)
        self.times.append(lap_time)
        print("Lap time:", lap_time)

    def total_time(self):
        # Calcula y devuelve el tiempo total transcurrido
        total_hours = sum([time[0] for time in self.times])
        total_minutes = sum([time[1] for time in self.times])
        total_seconds = sum([time[2] for time in self.times])
        return total_hours, total_minutes, total_seconds

    def printReport(self):
        # Imprime el reporte de los tiempos de vuelta y el tiempo total
        print("Times of each lap:", self.times)
        print("Total time:", self.total_time())


class Temporizador(Clock):
    def __init__(self, duration, is_24hour=True):
        hour, minute, second = duration
        super().__init__(hour, minute, second, is_24hour)
        self.duration = duration

    def countdown(self):
        # Realiza la cuenta regresiva del temporizador
        while self.hour != 0 or self.minute != 0 or self.second != 0:
            self.onesec()
            self.update()
            self.printTimeRemaining()
            if self.running == False:
                break
        print("Tiempo Cumplido")

    def printTimeRemaining(self):
        # Imprime el tiempo restante del temporizador
        print("Time remaining:", "{:02d}:{:02d}:{:02d}".format(self.hour, self.minute, self.second))

# Ejemplo de uso del Cronometro
cronometro = Cronometro()
cronometro.work()

# Ejemplo de uso del Temporizador
duracion = (0, 0, 5)  # Temporizador de 5 segundos
temporizador = Temporizador(duracion)
temporizador.countdown()
