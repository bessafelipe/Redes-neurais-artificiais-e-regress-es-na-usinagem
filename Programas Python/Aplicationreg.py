from Interfacereg import *
from Modeloclasseranreg import *
from Bancodedadosreg import *
import os
import matplotlib.lines as mlines
import threading 
from tkinter import filedialog
from pandastable import Table,config
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import shutil
from PIL import ImageTk,Image
import subprocess

initDB()

def atualizar_listbox():
    p_inicial.tree.delete(*p_inicial.tree.get_children())
    id2 = p_inicial.tree.insert("", 1,  text="Força")
    id3 = p_inicial.tree.insert("", 1,  text="Rugosidade")
    for r in view_força():
        p_inicial.tree.insert(id2, "end",  text=r[0], values=(r[2],r[4],r[3]))
    for r in view_rugosidade():
        p_inicial.tree.insert(id3, "end",iid=(r[0],r[2]),  text=r[0], values=(r[2],r[4],r[3]))
    
def selecionar():
    selected=p_inicial.tree.item(p_inicial.tree.focus())
    referencia=selected['text']
    tipo=selected['values'][0]
    
    return(referencia,tipo)


def dados_modelo():
    
    model=modelo(selecionar()[0],selecionar()[1])
    model.variaveis_exibir()
    pm=p_modelos
    
    pm.t_grandeza.delete("1.0","end-1c")
    pm.t_tipo.delete("1.0","end-1c")
    pm.t_referencia.delete("1.0","end-1c")
    pm.t_material.delete("1.0","end-1c")
    pm.t_ferramenta.delete("1.0","end-1c")
    pm.t_numero.delete("1.0","end-1c")
    pm.t_observacoes.delete("1.0","end-1c")
    
    pm.t_grandeza.insert('end',model.grandeza)
    pm.t_tipo.insert('end',model.tipo)
    pm.t_referencia.insert('end',model.referencia)
    pm.t_material.insert('end',model.material)
    pm.t_ferramenta.insert('end',model.ferramenta)
    pm.t_numero.insert('end',model.numero)
    pm.t_observacoes.insert('end',model.observacoes)
    
    imageg1=[]
    graficog1=[]
    pm.graficog1.destroy()
    imageg1=Image.open('Arquivos\Gráficos/'+model.name+'_teste.png')
    imageg1 = imageg1.resize((320, 270), Image.ANTIALIAS)
    graficog1 = ImageTk.PhotoImage(imageg1)

    pm.graficog1 = tk.Label(pm.f_graficos, image=graficog1)
    pm.graficog1.image=graficog1
    pm.graficog1.grid(padx='5', pady='5')

    imageg2=[]
    graficog2=[]
    pm.graficog2.destroy()
    imageg2=Image.open('Arquivos\Gráficos/'+model.name+'_treino.png')
    imageg2 = imageg2.resize((320, 270), Image.ANTIALIAS)
    graficog2 = ImageTk.PhotoImage(imageg2)

    pm.graficog2 = tk.Label(pm.f_graficos, image=graficog2)
    pm.graficog2.image=graficog2
    pm.graficog2.grid(padx='5', pady='5')
    
    options = config.load_options()
    options = {'font': 'Times New Roman',
               'fontsize': 12, 'cellwidth': 100}
    
    pt = Table(pm.f_tab1, dataframe=model.teste,width=400, height=500)
    config.apply_options(options, pt)
    pt.show()
    
    pt2=Table(pm.f_tab2, dataframe=model.treino,width=400, height=500)
    config.apply_options(options, pt2)
    pt2.show()
        
    
    for reg in ['RL','RP2','RP3','RP4','RN']:
        
        pm.e_velocidade[reg].delete(0,"end")
        pm.e_avanco[reg].delete(0,"end")
        pm.e_profundidade[reg].delete(0,"end")
        
        pm.t_saida[reg].delete("1.0","end-1c")
        
        pm.t_erm_test[reg].delete("1.0","end-1c")
        pm.t_cd_test[reg].delete("1.0","end-1c")
        pm.t_cc_test[reg].delete("1.0","end-1c")
        pm.t_mse_test[reg].delete("1.0","end-1c")
        pm.t_rmse_test[reg].delete("1.0","end-1c")
        
        pm.t_erm_test[reg].insert('end',model.errorel_test[reg])
        pm.t_cd_test[reg].insert('end',model.det_test[reg])
        pm.t_cc_test[reg].insert('end',model.cor_test[reg])
        pm.t_mse_test[reg].insert('end',model.mse_test[reg])
        pm.t_rmse_test[reg].insert('end',model.rmse_test[reg])
        
        pm.t_erm_train[reg].delete("1.0","end-1c")
        pm.t_cd_train[reg].delete("1.0","end-1c")
        pm.t_cc_train[reg].delete("1.0","end-1c")
        pm.t_mse_train[reg].delete("1.0","end-1c")
        pm.t_rmse_train[reg].delete("1.0","end-1c")
        
        pm.t_erm_train[reg].insert('end',model.errorel_train[reg])
        pm.t_cd_train[reg].insert('end',model.det_train[reg])
        pm.t_cc_train[reg].insert('end',model.cor_train[reg])
        pm.t_mse_train[reg].insert('end',model.mse_train[reg])
        pm.t_rmse_train[reg].insert('end',model.rmse_train[reg])
        
        pm.l_u_velocidade[reg].configure(text=model.u_velocidade)
        pm.l_u_avanco[reg].configure(text=model.u_avanco)
        pm.l_u_profundidade[reg].configure(text=model.u_profundidade)
        pm.l_saida[reg].configure(text=model.grandeza+':')
        pm.l_u_saida[reg].configure(text=model.u_saida)
        
        
        image=Image.open( 'Arquivos/'+reg+'/Gráficos/'+model.name+'_teste.png')
        image = image.resize((250, 250), Image.ANTIALIAS)
        grafico1 = ImageTk.PhotoImage(image)

        pm.grafico1[reg] = tk.Label(pm.f_erro[reg], image=grafico1)
        pm.grafico1[reg].image=grafico1
        pm.grafico1[reg].grid(padx='10', pady='10',row='1',column='0',rowspan='7')
        
        image2=Image.open( 'Arquivos/'+reg+'/Gráficos/'+model.name+'_treino.png')
        image2 = image2.resize((250, 250), Image.ANTIALIAS)
        grafico2 = ImageTk.PhotoImage(image2)

        pm.grafico2[reg] = tk.Label(pm.f_erro[reg], image=grafico2)
        pm.grafico2[reg].image=grafico2
        pm.grafico2[reg].grid(padx='10', pady='10',row='8',column='0',rowspan='7')
        
def previsao(reg):
    pm=p_modelos
    pm.t_saida[reg].delete("1.0","end-1c")

    model=modelo(selecionar()[0],selecionar()[1])
    model.carregar()
    
    f=float(pm.e_avanco[reg].get())
    a=float(pm.e_profundidade[reg].get())
    v=float(pm.e_velocidade[reg].get())
    força=round(model.pred(v,f,a)[reg],4)
    
    pm.t_saida[reg].insert('end',força)
    
    pm.e_avanco[reg].delete(0,"end")
    pm.e_profundidade[reg].delete(0,"end")
    pm.e_velocidade[reg].delete(0,"end")        
        
        
    



def criar():
    pn=p_novo

    
    referencia=pn.t_referencia.get("1.0","end-1c")
    grandeza=pn.grandeza_r.get()
    tipo=pn.t_tipo.get("1.0","end-1c")
    material=pn.t_material.get("1.0","end-1c")
    ferramenta=pn.t_ferramenta.get("1.0","end-1c")
    numero=pn.t_numero.get("1.0","end-1c")
    observacoes=pn.t_observacoes.get("1.0","end-1c")
    u_velocidade=pn.t_u_velocidade.get("1.0","end-1c")
    u_avanco=pn.t_u_avanco.get("1.0","end-1c")
    u_profundidade=pn.t_u_profundidade.get("1.0","end-1c")
    u_saida=pn.t_u_saida.get("1.0","end-1c")

    insert(referencia,grandeza,tipo,ferramenta,material,numero,observacoes,u_velocidade,
           u_avanco,u_profundidade,u_saida)
    data = {grandeza: [], 'n': [],'f': [],'a': []}
    df=pd.DataFrame(data)
    df.to_csv('Arquivos\Dados/'+referencia+'_'+tipo+'_dados.csv',
              index=False,sep=";")
    os.startfile('Arquivos\Dados/'+referencia+'_'+tipo+'_dados.csv')
    
    atualizar_listbox()
   
def exibir_editar():
    model=modelo(selecionar()[0],selecionar()[1])
    pe=p_editar
    
    pe.t_grandeza.delete("1.0","end-1c")
    pe.t_tipo.delete("1.0","end-1c")
    pe.t_referencia.delete("1.0","end-1c")
    pe.t_material.delete("1.0","end-1c")
    pe.t_ferramenta.delete("1.0","end-1c")
    pe.t_numero.delete("1.0","end-1c")
    pe.t_observacoes.delete("1.0","end-1c")
    pe.t_u_velocidade.delete("1.0","end-1c")
    pe.t_u_avanco.delete("1.0","end-1c")
    pe.t_u_profundidade.delete("1.0","end-1c")
    pe.t_u_saida.delete("1.0","end-1c") 
    
    pe.t_grandeza.insert('end',model.grandeza)
    pe.t_tipo.insert('end',model.tipo)
    pe.t_referencia.insert('end',model.referencia)
    pe.t_material.insert('end',model.material)
    pe.t_ferramenta.insert('end',model.ferramenta)
    pe.t_numero.insert('end',model.numero)
    pe.t_observacoes.insert('end',model.observacoes)
    pe.t_u_velocidade.insert('end',model.u_velocidade)
    pe.t_u_avanco.insert('end',model.u_avanco)
    pe.t_u_profundidade.insert('end',model.u_profundidade)
    pe.t_u_saida.insert('end',model.u_saida)
    
    pe.l_u_saida.configure(text=model.grandeza+':')
    
def editar():   
    referencia=pe.t_referencia.get("1.0","end-1c")
    grandeza=pe.t_grandeza.get("1.0","end-1c")
    tipo=pe.t_tipo.get("1.0","end-1c")
    material=pe.t_material.get("1.0","end-1c")
    ferramenta=pe.t_ferramenta.get("1.0","end-1c")
    numero=pe.t_numero.get("1.0","end-1c")
    observacoes=pe.t_observacoes.get("1.0","end-1c")
    u_velocidade=pe.t_u_velocidade.get("1.0","end-1c")
    u_avanco=pe.t_u_avanco.get("1.0","end-1c")
    u_profundidade=pe.t_u_profundidade.get("1.0","end-1c")
    u_saida=pe.t_u_saida.get("1.0","end-1c")

    os.rename(r'Arquivos\Dados/'+selecionar()[0]+'_'+selecionar()[1]+'_dados.csv',
              r'Arquivos\Dados/'+referencia+'_'+tipo+'_dados.csv')
    
    os.rename(r'Arquivos\Artigos/'+selecionar()[0]+'_'+selecionar()[1]+'_artigo.pdf',
              r'Arquivos\Artigos/'+referencia+'_'+tipo+'_artigo.pdf')

    update(referencia,tipo,ferramenta,material,numero,observacoes,selecionar()[0],selecionar()[1])

    atualizar_listbox()
    
    os.startfile('Arquivos\Dados/'+referencia+'_'+tipo+'_dados.csv')
    
def popup(modo): 
    global app2, pi,pc,pp
    app2=SampleApp2()
    pi=app2.frames[popup_inserir.__name__]
    pc=app2.frames[popup_criando.__name__]
    pp=app2.frames[popup_pronto.__name__]
    
    if modo=='edit':
        pi.b_ok.configure(command=lambda: [funçao(1),])
    elif modo=='new':
        pi.b_ok.configure(command=lambda: [funçao(2),])
    else:
        print('Erro ')
    
    app2.mainloop()
    
def funçao(modo):  
    pi.controller.show_frame('popup_criando')
    pc.bar.start()
    
    if modo==1:
        referencia=pe.t_referencia.get("1.0","end-1c")
        tipo=pe.t_tipo.get("1.0","end-1c")
        pe.t_grandeza.delete("1.0","end-1c")
        pe.t_tipo.delete("1.0","end-1c")
        pe.t_referencia.delete("1.0","end-1c")
        pe.t_material.delete("1.0","end-1c")
        pe.t_ferramenta.delete("1.0","end-1c")
        pe.t_numero.delete("1.0","end-1c")
        pe.t_observacoes.delete("1.0","end-1c")
        pe.t_u_velocidade.delete("1.0","end-1c")
        pe.t_u_avanco.delete("1.0","end-1c")
        pe.t_u_profundidade.delete("1.0","end-1c")
        pe.t_u_saida.delete("1.0","end-1c")
    elif modo==2:
        referencia=pn.t_referencia.get("1.0","end-1c")
        tipo=pn.t_tipo.get("1.0","end-1c")

        pn.t_tipo.delete("1.0","end-1c")
        pn.t_referencia.delete("1.0","end-1c")
        pn.t_material.delete("1.0","end-1c")
        pn.t_ferramenta.delete("1.0","end-1c")
        pn.t_numero.delete("1.0","end-1c")
        pn.t_observacoes.delete("1.0","end-1c")
        pn.t_u_velocidade.delete("1.0","end-1c")
        pn.t_u_avanco.delete("1.0","end-1c")
        pn.t_u_profundidade.delete("1.0","end-1c")
        pn.t_u_saida.delete("1.0","end-1c")
    else:
        print('erro')
        
    t=threading.Thread(target=criando_modelos,args=(referencia,tipo)).start()   
    
def criando_modelos(referencia,tipo):
    model=modelo(referencia,tipo)
    cv=int(pi.e_validacao.get())
    ite=int(pi.e_iteracoes.get())
    pi.e_validacao.delete(0,"end")
    pi.e_iteracoes.delete(0,"end")
    model.criar_modelos(cv,ite)
    
    pc.bar.stop()
    pc.controller.show_frame('popup_pronto')
    p_novo.b_dados_experimentais.configure(state='disabled'),
    p_novo.controller.show_frame("p_inicial")
    pp.b_fechar.configure(command=lambda: app2.destroy())
    

def anexar_artigo(): 
    referencia=pn.t_referencia.get("1.0","end-1c")
    tipo=pn.t_tipo.get("1.0","end-1c")
    source =filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("pdf files","*.pdf"),("all files","*.*")))
     
    destination ='Arquivos/Artigos/'+referencia+'_'+tipo+'_artigo.pdf'
    dest = shutil.copy(source, destination)
    p_novo.b_dados_experimentais.configure(state='normal')
    
def deletar():
    delete(selecionar()[0],selecionar()[1])
    v=[]
    v.append(lambda: os.remove('Arquivos\Dados/'+selecionar()[0]+'_'+selecionar()[1]+'_dados.csv'))
    v.append(lambda: os.remove('Arquivos\Artigos/'+selecionar()[0]+'_'+selecionar()[1]+'_artigo.pdf'))
    v.append(lambda: os.remove('Arquivos\Gráficos/'+selecionar()[0]+'_'+selecionar()[1]+'_teste.png'))
    v.append(lambda: os.remove('Arquivos\Gráficos/'+selecionar()[0]+'_'+selecionar()[1]+'_treino.png'))
    v.append(lambda: os.remove('Arquivos\Relatórios/'+selecionar()[0]+'_'+selecionar()[1]+'_relatorio.docx'))
    

    v.append(lambda: os.remove('Arquivos\RN/Erros/'+selecionar()[0]+'_'+selecionar()[1]+'_erros.json'))
    v.append(lambda: os.remove('Arquivos\RN/Gráficos/'+selecionar()[0]+'_'+selecionar()[1]+'_teste.png'))
    v.append(lambda: os.remove('Arquivos\RN/Gráficos/'+selecionar()[0]+'_'+selecionar()[1]+'_treino.png'))
    v.append(lambda: os.remove('Arquivos\RN/Modelos/'+selecionar()[0]+'_'+selecionar()[1]+'_modelo.mdl'))
    
    v.append(lambda: os.remove('Arquivos\RL/Erros/'+selecionar()[0]+'_'+selecionar()[1]+'_erros.json'))
    v.append(lambda: os.remove('Arquivos\RL/Gráficos/'+selecionar()[0]+'_'+selecionar()[1]+'_teste.png'))
    v.append(lambda: os.remove('Arquivos\RL/Gráficos/'+selecionar()[0]+'_'+selecionar()[1]+'_treino.png'))
    v.append(lambda: os.remove('Arquivos\RL/Modelos/'+selecionar()[0]+'_'+selecionar()[1]+'_modelo.mdl'))
    
    v.append(lambda: os.remove('Arquivos\RP2/Erros/'+selecionar()[0]+'_'+selecionar()[1]+'_erros.json'))
    v.append(lambda: os.remove('Arquivos\RP2/Gráficos/'+selecionar()[0]+'_'+selecionar()[1]+'_teste.png'))
    v.append(lambda: os.remove('Arquivos\RP2/Gráficos/'+selecionar()[0]+'_'+selecionar()[1]+'_treino.png'))
    v.append(lambda: os.remove('Arquivos\RP2/Modelos/'+selecionar()[0]+'_'+selecionar()[1]+'_modelo.mdl'))
    
    v.append(lambda: os.remove('Arquivos\RP3/Erros/'+selecionar()[0]+'_'+selecionar()[1]+'_erros.json'))
    v.append(lambda: os.remove('Arquivos\RP3/Gráficos/'+selecionar()[0]+'_'+selecionar()[1]+'_teste.png'))
    v.append(lambda: os.remove('Arquivos\RP3/Gráficos/'+selecionar()[0]+'_'+selecionar()[1]+'_treino.png'))
    v.append(lambda: os.remove('Arquivos\RP3/Modelos/'+selecionar()[0]+'_'+selecionar()[1]+'_modelo.mdl'))
    
    v.append(lambda: os.remove('Arquivos\RP4/Erros/'+selecionar()[0]+'_'+selecionar()[1]+'_erros.json'))
    v.append(lambda: os.remove('Arquivos\RP4/Gráficos/'+selecionar()[0]+'_'+selecionar()[1]+'_teste.png'))
    v.append(lambda: os.remove('Arquivos\RP4/Gráficos/'+selecionar()[0]+'_'+selecionar()[1]+'_treino.png'))
    v.append(lambda: os.remove('Arquivos\RP4/Modelos/'+selecionar()[0]+'_'+selecionar()[1]+'_modelo.mdl'))
    
    v.append(lambda: os.remove('Arquivos\RN/Pesos/'+selecionar()[0]+'_'+selecionar()[1]+'_pesos.txt'))
    v.append(lambda: os.remove('Arquivos\RN/Parâmetros/'+selecionar()[0]+'_'+selecionar()[1]+'_parametros.json'))

    v.append(lambda: os.remove('Arquivos\RL/Coeficientes/'+selecionar()[0]+'_'+selecionar()[1]+'_coeficientes.txt'))
    
    v.append(lambda: os.remove('Arquivos\RP2/Coeficientes/'+selecionar()[0]+'_'+selecionar()[1]+'_coeficientes.txt'))
    
    v.append(lambda: os.remove('Arquivos\RP3/Coeficientes/'+selecionar()[0]+'_'+selecionar()[1]+'_coeficientes.txt'))
    
    v.append(lambda: os.remove('Arquivos\RP4/Coeficientes/'+selecionar()[0]+'_'+selecionar()[1]+'_coeficientes.txt'))
    
    for func in v:
        try:
            func()
        except:
            print("Arquivo nao encontrado") 
        
    atualizar_listbox()
    
    
def abrir_artigo():
    os.startfile('Arquivos\Artigos/'+selecionar()[0]+'_'+selecionar()[1]+'_artigo.pdf')
    

def abrir_relatorio():
    os.startfile(r'Arquivos\Relatórios/'+selecionar()[0]+'_'+selecionar()[1]+'_relatorio.docx')
    
def abrir_arquivos():
    os.startfile(r'Arquivos')
    
def abrir_iteracoes():
    os.startfile(r'Arquivos\RN/Iterações/'+selecionar()[0]+'_'+selecionar()[1]+'_iteraçoes.txt')
    
def abrir_pesos():
    os.startfile(r'Arquivos\RN/Pesos/'+selecionar()[0]+'_'+selecionar()[1]+'_pesos.txt')
    
def abrir_parametros():
    subprocess.call(['notepad.exe',r'Arquivos\RN/Parâmetros/'+selecionar()[0]+'_'+selecionar()[1]+'_parametros.json'])
    
def abrir_coeficientes(reg):
    os.startfile(r"Arquivos\\"+reg+'/Coeficientes/'+selecionar()[0]+'_'+selecionar()[1]+'_coeficientes.txt')
    

if __name__ == "__main__":
    
    initDB()
    app = SampleApp()

    menubar = tk.Menu(app)

    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label="Open")
    filemenu.add_command(label="Save")
    filemenu.add_command(label="Exit")
    
    menubar.add_cascade(label="File", menu=filemenu)
    
    app.config(menu=menubar)

    app.title("Rede neural no torneamento")
    app.geometry('1000x650+50+0')
    p_inicial=app.frames[p_inicial.__name__]
    p_novo=app.frames[p_novo.__name__]
    pn=p_novo
    p_editar=app.frames[p_editar.__name__]
    pe=p_editar
    p_modelos=app.frames[p_modelos.__name__]
    pm=p_modelos
    
    atualizar_listbox()

    p_inicial.button_abrir.configure(command=lambda: [p_inicial.controller.show_frame("p_modelos"),
                                                      dados_modelo()])
    p_inicial.button_novo.configure(command=lambda: p_inicial.controller.show_frame("p_novo"))
    
    p_inicial.button_editar.configure(command=lambda: [p_inicial.controller.show_frame("p_editar"),
                                                       exibir_editar()])
    p_inicial.button_deletar.configure(command=lambda:deletar())
    
    p_novo.b_anexar_artigo.configure(command=lambda: anexar_artigo())
    p_novo.b_dados_experimentais.configure(command=lambda: [criar(),popup('new')])
    p_novo.b_voltar.configure(command=lambda:p_novo.controller.show_frame("p_inicial"))
    
    p_editar.b_atualizar.configure(command=lambda:[editar(),
                                                   popup('edit')])
    p_editar.b_voltar.configure(command=lambda:p_novo.controller.show_frame("p_inicial"))

    p_modelos.b_previsao['RN'].configure(command=lambda: previsao('RN'))
    p_modelos.b_iteracoes.configure(command=lambda: abrir_iteracoes())
    p_modelos.b_pesos.configure(command=lambda: abrir_pesos())
    p_modelos.b_parametros.configure(command=lambda: abrir_parametros())
    p_modelos.b_previsao['RL'].configure(command=lambda: previsao('RL'))
    p_modelos.b_coeficientes['RL'].configure(command=lambda: abrir_coeficientes('RL'))
    p_modelos.b_previsao['RP2'].configure(command=lambda: previsao('RP2'))
    p_modelos.b_coeficientes['RP2'].configure(command=lambda: abrir_coeficientes('RP2'))
    p_modelos.b_previsao['RP3'].configure(command=lambda: previsao('RP3'))
    p_modelos.b_coeficientes['RP3'].configure(command=lambda: abrir_coeficientes('RP3'))
    p_modelos.b_previsao['RP4'].configure(command=lambda: previsao('RP4'))
    p_modelos.b_coeficientes['RP4'].configure(command=lambda: abrir_coeficientes('RP4'))
    p_modelos.b_artigo.configure(command=lambda:abrir_artigo())
    p_modelos.b_relatorio.configure(command=lambda:abrir_relatorio())
    p_modelos.b_arquivos.configure(command=lambda:abrir_arquivos())
    p_modelos.b_voltar.configure(command=lambda:p_modelos.controller.show_frame("p_inicial"))
    
        
    '''
    Pagina1.button_novo.configure(command=lambda: [Pagina1.controller.show_frame("Pagina5"),])
    Pagina1.button_editar.configure(command=lambda: [selecionar(),
                                                Pagina1.controller.show_frame("Pagina4"),
                                                exibir_editar()])
    Pagina1.button_deletar.configure(command=lambda: deletar())

    Pagina3.button_4.configure(command=lambda: [Pagina3.controller.show_frame("Pagina2"),])
    
    Pagina4.button_1.configure(command=lambda: [Pagina4.controller.show_frame("Pagina1"),
                                                editar(),
                                                appp('edit'),])
    Pagina4.button_4.configure(command=lambda: Pagina5.controller.show_frame("Pagina1"))
   
    Pagina5.button_2.configure(command=lambda: [Pagina5.button_2.configure(state='disabled'),
                                                Pagina5.controller.show_frame("Pagina1"),
                                                criar(),
                                                appp('new')])
    Pagina5.button_1.configure(command=lambda: [anexar_artigo(),])
    Pagina5.button_4.configure(command=lambda: Pagina5.controller.show_frame("Pagina1"))
    
    Pagina6.button.configure(command=lambda: Pagina6.controller.show_frame("Pagina2"))
    '''
    app.mainloop()