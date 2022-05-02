import sqlite3 as sql

class TransactionObject():
    database    = "Arquivos/dados_estudo.db"
    conn        = None
    cur         = None
    connected   = False

    def connect(self):
        TransactionObject.conn      = sql.connect(TransactionObject.database)
        TransactionObject.cur       = TransactionObject.conn.cursor()
        TransactionObject.connected = True

    def disconnect(self):
        TransactionObject.conn.close()
        TransactionObject.connected = False

    def execute(self, sql, parms = None):
        if TransactionObject.connected:
            if parms == None:
                TransactionObject.cur.execute(sql)
            else:
                TransactionObject.cur.execute(sql, parms)
            return True
        else:
            return False

    def fetchall(self):
        return TransactionObject.cur.fetchall()

    def persist(self):
        if TransactionObject.connected:
            TransactionObject.conn.commit()
            return True
        else:
            return False

def initDB():
    trans = TransactionObject()
    trans.connect()
    trans.execute("""
        CREATE TABLE IF NOT EXISTS dados_estudo (
        referencia TEXT NOT NULL,
        grandeza TEXT NOT NULL,
        tipo TEXT NOT NULL,
        ferramenta TEXT NOT NULL,
        material TEXT NOT NULL,
        numero INTEGER,
        observaçoes TEXT NOT NULL,
        uvelocidade TEXT,
        uavanco TEXT,
        uprofundidade TEXT,
        usaida TEXT
        );
        """)

    trans.persist()
    trans.disconnect()

def insert(referencia,grandeza,tipo,ferramenta,material,numero,observaçoes,u_velocidade,u_avanco,u_profundidade,u_saida):
    trans = TransactionObject()
    trans.connect()
    trans.execute("INSERT INTO dados_estudo VALUES(?,?,?,?,?,?,?,?,?,?,?)", (referencia,grandeza,tipo,ferramenta,material,numero,observaçoes,u_velocidade,u_avanco,u_profundidade,u_saida))
    trans.persist()
    trans.disconnect()


def view_força():
    trans = TransactionObject()
    trans.connect()
    trans.execute("SELECT * FROM dados_estudo WHERE grandeza=?",('Força',))
    rows = trans.fetchall()
    trans.disconnect()
    return rows

def view_rugosidade():
    trans = TransactionObject()
    trans.connect()
    trans.execute("SELECT * FROM dados_estudo WHERE grandeza=?",('Rugosidade',))
    rows = trans.fetchall()
    trans.disconnect()
    return rows

def delete(name,tipo):
    trans = TransactionObject()
    trans.connect()
    trans.execute("DELETE FROM dados_estudo WHERE referencia = ? AND tipo=?", (name,tipo))
    trans.persist()
    trans.disconnect()
    
def update(referencia_new,tipo,ferramenta,material,numero,observaçoes,referencia_old,tipo_old):
    trans = TransactionObject()
    trans.connect()
    trans.execute("""
                  UPDATE dados_estudo SET 
                  referencia =?,
                  tipo=?,
                  ferramenta=?,
                  material=?,
                  numero=?,
                  observaçoes=?
                  WHERE referencia = ? AND tipo=?""",(referencia_new,tipo,ferramenta,material,numero,observaçoes,referencia_old,tipo_old))
    trans.persist()
    trans.disconnect()
    
def view_one(name,tipo):
    trans = TransactionObject()
    trans.connect()
    trans.execute("SELECT * FROM dados_estudo WHERE referencia=? AND tipo=?", (name,tipo))
    rows = trans.fetchall()
    trans.disconnect()
    return rows