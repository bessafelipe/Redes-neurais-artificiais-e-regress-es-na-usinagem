import tkinter as tk
import tkinter.ttk as ttk

fonte=('Segoe UI',10,'bold')

class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        container = ttk.Frame(self)
        container.pack(side="right", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (p_inicial,p_modelos,p_editar,p_novo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("p_inicial")
        
    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

class p_inicial(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)

        self.controller = controller

        self.button_abrir = ttk.Button(self,text='Abrir')
        self.button_abrir.grid(column='0', row='4',pady=10)
        
        self.button_novo = ttk.Button(self,text='Novo')
        self.button_novo.grid(column='1', row='4',pady=10)
        
        self.button_editar = ttk.Button(self,text='Editar')
        self.button_editar.grid(column='2', row='4',pady=10)
        
        self.button_deletar = ttk.Button(self,text='Deletar')
        self.button_deletar.grid(column='3', row='4',pady=10)
      
        self.tree = ttk.Treeview(self,height='27')
        self.tree["columns"]=("one","two","three")
        self.tree.column('#0', width=250 )
        self.tree.heading("#0", text="Referência")
        self.tree.column("one", width=100 )
        self.tree.column("two", width=250)
        self.tree.column("three", width=250)
        self.tree.heading("one", text="Tipo")
        self.tree.heading("two", text="Material")
        self.tree.heading("three", text="Ferramenta")
        self.tree.grid(column='0', columnspan='4', row='3',padx=10,pady=10)
 
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=1)
        
class p_modelos(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        
################################### Abas #################################
        
        self.note=ttk.Notebook(self)
        self.aba={}
        
        self.rede_neural=ttk.Frame(self.note)
        for reg in ['geral','dados','RN','RL','RP2','RP3','RP4']: 
            self.aba[reg]=ttk.Frame(self.note)


        self.note.add(self.aba['geral'],text=f'{"Geral": ^20s}')
        self.note.add(self.aba['dados'],text=f'{"Dados de treino e teste": ^30s}')
        self.note.add(self.aba['RN'],text=f'{"Rede Neural": ^30s}')
        self.note.add(self.aba['RL'],text=f'{"Regressão Linear": ^30s}')
        self.note.add(self.aba['RP2'],text=f'{"Regressão Polinomial 2": ^30s}')
        self.note.add(self.aba['RP3'],text=f'{"Regressão Polinomial 3": ^30s}')
        self.note.add(self.aba['RP4'],text=f'{"Regressão Polinomial 4": ^30s}')
   
        self.note.pack(expand=1, fill='both')
        
        self.f_button_geral=ttk.Frame(self)
        self.f_button_geral.pack(expand=1, fill='both')
        self.b_arquivos=ttk.Button(self.f_button_geral,text='Arquivos')
        self.b_arquivos.pack(side=tk.LEFT,expand=1, fill='both')
        self.b_artigo=ttk.Button(self.f_button_geral,text='Artigo')
        self.b_artigo.pack(side=tk.LEFT,expand=1, fill='both')
        self.b_relatorio=ttk.Button(self.f_button_geral,text='Relatório')
        self.b_relatorio.pack(side=tk.LEFT,expand=1, fill='both')
        self.b_voltar=ttk.Button(self.f_button_geral,text='Voltar')
        self.b_voltar.pack(side=tk.LEFT,expand=1, fill='both')
        


       
        
#################################### Aba geral ##########################
        
        self.f_info=ttk.Frame(self.aba['geral'])
        self.f_info.grid()
   
        self.aba['geral'].grid_columnconfigure(0, weight=2)
        self.aba['geral'].grid_columnconfigure(1, weight=1)
        
        self.l_informaçoes_do_estudo=ttk.Label(self.f_info,text="Informações do estudo", font=fonte)
        self.l_informaçoes_do_estudo.grid(padx='10', pady='10',columnspan='2')
        
        self.l_grandeza = ttk.Label(self.f_info,text='Grandeza:')
        self.l_grandeza.grid(padx='10', pady='10', row='1', sticky='e',column='0')
        
        self.t_grandeza = tk.Text(self.f_info, height='1',  width='50')
        self.t_grandeza.grid(column='1', padx='10', pady='10', row='1')
        
        self.l_tipo=ttk.Label(self.f_info,text='Tipo:')
        self.l_tipo.grid(column='0', padx='10', pady='10', row='2',sticky='ne')
        
        self.t_tipo=tk.Text(self.f_info,height='1', undo='false', width='50')
        self.t_tipo.grid(column='1', padx='10', pady='10', row='2')
        
        self.l_referencia = ttk.Label(self.f_info,text='Referência:')
        self.l_referencia.grid(column='0', padx='10', pady='10', row='3', sticky='ne')
        
        self.t_referencia = tk.Text(self.f_info,height='1', undo='false', width='50')
        self.t_referencia.grid(column='1', padx='10', pady='10', row='3')
        
        self.l_material = ttk.Label(self.f_info,text='Material:')
        self.l_material.grid(column='0', padx='10', pady='10', row='4', sticky='e')
        
        self.t_material = tk.Text(self.f_info,height='1', width='50')
        self.t_material.grid(column='1', padx='10', pady='10', row='4')
        
        self.l_ferramenta = ttk.Label(self.f_info,text='Ferramenta de corte:')
        self.l_ferramenta.grid(column='0', padx='10', pady='5', row='5', sticky='e')
        
        self.t_ferramenta = tk.Text(self.f_info,height='1', state='normal', width='50')
        self.t_ferramenta.grid(column='1', padx='10', pady='10', row='5')
        
        self.l_numero = ttk.Label(self.f_info,text='N° de experimentos:')
        self.l_numero.grid(column='0', padx='10', pady='10', row='6', sticky='e')
       
        self.t_numero = tk.Text(self.f_info,height='1', width='50')
        self.t_numero.grid(column='1', padx='10', pady='10', row='6')
        
        self.l_observacoes = ttk.Label(self.f_info,text='Observações:')
        self.l_observacoes.grid(column='0', padx='10', pady='10', row='7', sticky='ne')
        
        self.t_observacoes = tk.Text(self.f_info,height='10', width='50')
        self.t_observacoes.grid(column='1', padx='10', pady='10', row='7')
            
        self.f_graficos=ttk.Frame(self.aba['geral'])
        self.f_graficos.grid(column='1',row='0')
        
        self.graficog1=ttk.Label(self.f_graficos)
        self.graficog2=ttk.Label(self.f_graficos)
        
####################################  Aba rede neural ###############################
        
        self.f_estimativa={}
        self.l_estimativa={}
        self.l_dados_de_entrada={}
        self.l_velocidade={}
        self.l_avanco={}
        self.l_profundidade={}
        self.e_velocidade={}
        self.e_avanco={}
        self.e_profundidade={}
        self.l_u_velocidade={}
        self.l_u_avanco={}
        self.l_u_profundidade={}
        self.l_saida={}
        self.l_u_saida={}
        self.f_estimativa={}
        self.t_saida={}
        self.b_previsao={}
        self.separator={}
        
        self.f_erro={}
        self.l_metricas={}
        self.l_erm_test={}
        self.t_erm_test={}
        self.l_cd_test={}
        self.t_cd_test={}
        self.l_cc_test={}
        self.t_cc_test={}
        self.l_mse_test={}
        self.t_mse_test={}
        self.l_rmse_test={}
        self.t_rmse_test={}
        
        self.l_erm_train={}
        self.t_erm_train={}
        self.l_cd_train={}
        self.t_cd_train={}
        self.l_cc_train={}
        self.t_cc_train={}
        self.l_mse_train={}
        self.t_mse_train={}
        self.l_rmse_train={}
        self.t_rmse_train={}
        
        self.grafico1={}
        self.grafico2={}
        
        self.f_button={}
    
        for reg in ['RN','RL','RP2','RP3','RP4']:
        
            self.f_estimativa[reg]=ttk.Frame(self.aba[reg])
            self.f_estimativa[reg].grid( sticky='w')
            
            self.l_estimativa[reg] = ttk.Label(self.f_estimativa[reg],text='Estimativa',font=fonte)
            self.l_estimativa[reg].grid(column='0', padx='10', pady='10', row='0',columnspan='2')
            
            self.l_dados_de_entrada[reg] = ttk.Label(self.f_estimativa[reg],text='Dados de entrada:')
            self.l_dados_de_entrada[reg].grid(column='0', padx='10', pady='10', row='1')
            
            self.l_velocidade[reg] = ttk.Label(self.f_estimativa[reg],text='Velocidade:')
            self.l_velocidade[reg].grid(column='0', padx='10', pady='10', row='2', sticky='e')
            
            self.l_avanco[reg] = ttk.Label(self.f_estimativa[reg],text='Avanço:')
            self.l_avanco[reg].grid(column='0', padx='10', pady='10', row='3', sticky='e')
            
            self.l_profundidade[reg] = ttk.Label(self.f_estimativa[reg],text='Profundidade de corte:')
            self.l_profundidade[reg].grid(column='0', padx='10', pady='10', row='4', sticky='e')
            
            self.e_velocidade[reg] = tk.Entry(self.f_estimativa[reg],width='15')
            self.e_velocidade[reg].grid(column='1', padx='10', pady='10', row='2')
            
            self.l_u_velocidade[reg] = ttk.Label(self.f_estimativa[reg],text='')
            self.l_u_velocidade[reg].grid(column='2', padx='10', pady='10', row='2', sticky='nw')
            
            self.e_avanco[reg] = tk.Entry(self.f_estimativa[reg],width='15')
            self.e_avanco[reg].grid(column='1', padx='10', pady='10', row='3')
            
            self.l_u_avanco[reg] = ttk.Label(self.f_estimativa[reg],text='')
            self.l_u_avanco[reg].grid(column='2', padx='10', pady='10', row='3', sticky='nw')
            
            self.e_profundidade[reg] = tk.Entry(self.f_estimativa[reg],width='15')
            self.e_profundidade[reg].grid(column='1', padx='10', pady='10', row='4')
            
            self.l_u_profundidade[reg] = ttk.Label(self.f_estimativa[reg],text='')
            self.l_u_profundidade[reg].grid(column='2', padx='10', pady='10', row='4', sticky='nw')
            
            self.l_saida[reg] = ttk.Label(self.f_estimativa[reg],text='Saída:')
            self.l_saida[reg].grid(column='0', padx='10', pady='10', row='6', sticky='ne')
            
            self.t_saida[reg] = tk.Text(self.f_estimativa[reg],height='1', width='11')
            self.t_saida[reg].grid(column='1', padx='10', pady='10', row='6', sticky='ne')
            
            self.l_u_saida[reg] = ttk.Label(self.f_estimativa[reg],text='')
            self.l_u_saida[reg].grid(column='2', padx='10', pady='10', row='6', sticky='nw')
            
            self.b_previsao[reg] = ttk.Button(self.f_estimativa[reg],text='Previsão')
            self.b_previsao[reg].grid(column='0', columnspan='2', padx='10', pady='10', row='5')
            
            self.separator[reg] = ttk.Separator(self.aba[reg],orient='vertical')
            self.separator[reg].grid(column='1', row='0', rowspan='2',padx='15', pady='10', sticky='ns')
                

            
            
            self.f_erro[reg]=ttk.Frame(self.aba[reg])
            self.f_erro[reg].grid(row='0',column='2',rowspan='2')
            
            self.l_metricas[reg]=ttk.Label(self.f_erro[reg],text="Métricas de performance",font=fonte)
            self.l_metricas[reg].grid(columnspan='3',padx='10', pady='10')
            
            
            
            self.l_erm_test[reg] = ttk.Label(self.f_erro[reg],text='Erro relativo médio:')
            self.l_erm_test[reg].grid(padx='5', pady='5', row='1', sticky='e',column='1')
            
            self.t_erm_test[reg] = tk.Text(self.f_erro[reg],height='1', width='10')
            self.t_erm_test[reg].grid(column='2', padx='5', pady='5', row='1')
            
            self.l_cd_test[reg] = ttk.Label(self.f_erro[reg],text='Coeficiente de determinaçao:')
            self.l_cd_test[reg].grid(column='1', padx='5', pady='5', row='2', sticky='e')
            
            self.t_cd_test[reg] = tk.Text(self.f_erro[reg],height='1', width='10')
            self.t_cd_test[reg].grid(column='2', padx='5', pady='5', row='2')
            
            self.l_cc_test[reg] = ttk.Label(self.f_erro[reg],text='Coeficiente de correlação:')
            self.l_cc_test[reg].grid(column='1', padx='5', pady='5', row='3', sticky='e')
            
            self.t_cc_test[reg]= tk.Text(self.f_erro[reg],height='1', width='10')
            self.t_cc_test[reg].grid(column='2', padx='5', pady='5', row='3')
            
            self.l_mse_test[reg] = ttk.Label(self.f_erro[reg],text='MSE:')
            self.l_mse_test[reg].grid(column='1', padx='5', pady='5', row='4', sticky='e')
            
            self.t_mse_test[reg] = tk.Text(self.f_erro[reg],height='1', width='10')
            self.t_mse_test[reg].grid(column='2', padx='5', pady='5', row='4')
            
            self.l_rmse_test[reg] = ttk.Label(self.f_erro[reg],text='RMSE:')
            self.l_rmse_test[reg].grid(column='1', padx='5', pady='5', row='5', sticky='e')
            
            self.t_rmse_test[reg] = tk.Text(self.f_erro[reg],height='1', width='10')
            self.t_rmse_test[reg].grid(column='2', padx='5', pady='5', row='5')
            
            
            
            self.l_erm_train[reg] = ttk.Label(self.f_erro[reg],text='Erro relativo médio:')
            self.l_erm_train[reg].grid(padx='5', pady='5', row='8', sticky='e',column='1')
            
            self.t_erm_train[reg] = tk.Text(self.f_erro[reg],height='1', width='10')
            self.t_erm_train[reg].grid(column='2', padx='5', pady='5', row='8')
            
            self.l_cd_train[reg] = ttk.Label(self.f_erro[reg],text='Coeficiente de determinaçao:')
            self.l_cd_train[reg].grid(column='1', padx='5', pady='5', row='9', sticky='e')
            
            self.t_cd_train[reg] = tk.Text(self.f_erro[reg],height='1', width='10')
            self.t_cd_train[reg].grid(column='2', padx='5', pady='5', row='9')
            
            self.l_cc_train[reg] = ttk.Label(self.f_erro[reg],text='Coeficiente de correlação:')
            self.l_cc_train[reg].grid(column='1', padx='5', pady='5', row='10', sticky='e')
            
            self.t_cc_train[reg]= tk.Text(self.f_erro[reg],height='1', width='10')
            self.t_cc_train[reg].grid(column='2', padx='5', pady='5', row='10')
            
            self.l_mse_train[reg] = ttk.Label(self.f_erro[reg],text='MSE:')
            self.l_mse_train[reg].grid(column='1', padx='5', pady='5', row='11', sticky='e')
            
            self.t_mse_train[reg] = tk.Text(self.f_erro[reg],height='1', width='10')
            self.t_mse_train[reg].grid(column='2', padx='5', pady='5', row='11')
            
            self.l_rmse_train[reg]= ttk.Label(self.f_erro[reg],text='RMSE:')
            self.l_rmse_train[reg].grid(column='1', padx='5', pady='5', row='12', sticky='e')
            
            self.t_rmse_train[reg]= tk.Text(self.f_erro[reg],height='1', width='10')
            self.t_rmse_train[reg].grid(column='2', padx='5', pady='5', row='12')
            
            self.f_button[reg]=ttk.Frame(self.aba[reg])
            self.f_button[reg].grid(column='0',row='1')
     
       
        self.b_parametros=ttk.Button(self.f_button['RN'],text='Parâmetros')
        self.b_parametros.grid(padx='10', pady='10')
  
        self.b_iteracoes=ttk.Button(self.f_button['RN'],text='Iterações')
        self.b_iteracoes.grid(padx='10', pady='10',column='0',row='1')
        
        self.b_pesos=ttk.Button(self.f_button['RN'],text='Pesos')
        self.b_pesos.grid(padx='10', pady='10',column='0',row='2')
        
        self.b_coeficientes={}
        
        for reg in ['RL','RP2','RP3','RP4']:
            self.b_coeficientes[reg]=ttk.Button(self.f_button[reg],text='Coeficientes')
            self.b_coeficientes[reg].pack()
                
        
################################ Aba dados de treino e teste #####################
        
        self.l_dados_teste=tk.Label(self.aba['dados'],text='Dados de teste',font=fonte)
        self.l_dados_teste.grid(row='0',column='0', padx='10', pady='10')
        
        self.l_dados_treino=tk.Label(self.aba['dados'],text='Dados de treino',font=fonte)
        self.l_dados_treino.grid(row='0',column='1', padx='10', pady='10')
        
        self.f_tab1=tk.Frame(self.aba['dados'])
        self.f_tab1.grid(row='1',column='0', padx='10', pady='10')
        
        self.f_tab2=tk.Frame(self.aba['dados'])
        self.f_tab2.grid(row='1',column='1', padx='10', pady='10')
        
        self.aba['dados'].grid_columnconfigure(0, weight=1)
        self.aba['dados'].grid_columnconfigure(1, weight=1)
   

          
class p_editar(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        
        self.f_editar=ttk.Frame(self)
        self.f_editar.place(anchor='c',relx=0.5,rely=0.4)
        
        self.l_edicao_de_modelo=ttk.Label(self.f_editar,text='Edição de modelo',font=fonte,anchor='c')
        self.l_edicao_de_modelo.grid(padx='10', pady='10', row='0', sticky='nesw',column='0',columnspan='2')
        
        self.l_grandeza = ttk.Label(self.f_editar,text='Grandeza:')
        self.l_grandeza.grid(padx='10', pady='10', row='1', sticky='e')
        
        self.t_grandeza = tk.Text(self.f_editar, height='1',  width='50')
        self.t_grandeza.grid(column='1', padx='10', pady='10', row='1')
        
        self.l_tipo = ttk.Label(self.f_editar,text='Tipo:')
        self.l_tipo.grid(padx='10', pady='10', row='2', sticky='e')
        
        self.t_tipo = tk.Text(self.f_editar, height='1',  width='50')
        self.t_tipo.grid(column='1', padx='10', pady='10', row='2')
        
        self.l_referencia = ttk.Label(self.f_editar,text='Referência:')
        self.l_referencia.grid(column='0', padx='10', pady='10', row='3', sticky='ne')
        
        self.t_referencia = tk.Text(self.f_editar,height='1', undo='false', width='50')
        self.t_referencia.grid(column='1', padx='10', pady='10', row='3')
        
        self.l_material = ttk.Label(self.f_editar,text='Material:')
        self.l_material.grid(column='0', padx='10', pady='10', row='4', sticky='e')
        
        self.t_material = tk.Text(self.f_editar,height='1', width='50')
        self.t_material.grid(column='1', padx='10', pady='10', row='4')
        
        self.l_ferramenta = ttk.Label(self.f_editar,text='Ferramenta de corte:')
        self.l_ferramenta.grid(column='0', padx='10', pady='10', row='5', sticky='e')
        
        self.t_ferramenta = tk.Text(self.f_editar,height='1', state='normal', width='50')
        self.t_ferramenta.grid(column='1', padx='10', pady='10', row='5')
        
        self.l_numero = ttk.Label(self.f_editar,text='N° de experimentos:')
        self.l_numero.grid(column='0', padx='10', pady='10', row='6', sticky='e')
       
        self.t_numero = tk.Text(self.f_editar,height='1', width='50')
        self.t_numero.grid(column='1', padx='10', pady='10', row='6')
        
        self.l_observacoes = ttk.Label(self.f_editar,text='Observações:')
        self.l_observacoes.grid(column='0', padx='10', pady='10', row='7', sticky='ne')
        
        self.t_observacoes = tk.Text(self.f_editar,height='10', width='50')
        self.t_observacoes.grid(column='1', padx='10', pady='10', row='7')
        
        self.l_unidades= tk.Label(self.f_editar, text= 'Unidades:')
        self.l_unidades.grid(column='2', padx='10', pady='10', row='1', sticky='nesw')
        
        self.l_u_velocidade = ttk.Label(self.f_editar,text='Velocidade:')
        self.l_u_velocidade.grid(column='2', padx='10', pady='10', row='2', sticky='ne')
        
        self.t_u_velocidade = tk.Text(self.f_editar,height='1', state='normal', width='15')
        self.t_u_velocidade.grid(column='3', padx='10', pady='10', row='2')
        
        self.l_u_avanco = ttk.Label(self.f_editar,text='Avanço:')
        self.l_u_avanco.grid(column='2', padx='10', pady='10', row='3', sticky='ne')
        
        self.t_u_avanco = tk.Text(self.f_editar,height='1', state='normal', width='15')
        self.t_u_avanco.grid(column='3', padx='10', pady='10', row='3')
        
        self.l_u_profundidade = ttk.Label(self.f_editar,text='Profundidade de corte:')
        self.l_u_profundidade.grid(column='2', padx='10', pady='10', row='4', sticky='ne')
        
        self.t_u_profundidade = tk.Text(self.f_editar,height='1', state='normal', width='15')
        self.t_u_profundidade.grid(column='3', padx='10', pady='10', row='4')
        
        self.l_u_saida = ttk.Label(self.f_editar,text='Força / Rugosidade:')
        self.l_u_saida.grid(column='2', padx='10', pady='10', row='5', sticky='ne')
        
        self.t_u_saida = tk.Text(self.f_editar,height='1', state='normal', width='15')
        self.t_u_saida.grid(column='3', padx='10', pady='10', row='5')
        
        self.b_atualizar = ttk.Button(self,text='Atualizar')
        self.b_atualizar.place(relx=0.5,rely=0.80,anchor='c')
        
        self.b_voltar = ttk.Button(self,text='Voltar')
        self.b_voltar.place(relx=0.9,rely=0.95)
        


class p_novo(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        
        
        
        self.l_criacao_de_modelo=ttk.Label(self,text='Criação de modelo',font=fonte,anchor='c')
        self.l_criacao_de_modelo.grid(padx='10', pady='10',columnspan='4')
        
        self.l_grandeza = ttk.Label(self,text='Grandeza:')
        self.l_grandeza.grid(padx='10', pady='10', row='1', sticky='e',column='0')
        
        self.grandeza_r = tk.StringVar()
        self.grandeza_r.set('Força')
        
        self.f_radiobutton=ttk.Frame(self)
        self.f_radiobutton.grid(row='1',column='1',sticky='w')
        self.radiobutton_força=ttk.Radiobutton(self.f_radiobutton,text="Força",
                                            variable=self.grandeza_r, value='Força')
        self.radiobutton_rugosidade=ttk.Radiobutton(self.f_radiobutton,text="Rugosidade",
                                            variable=self.grandeza_r, value='Rugosidade')
        self.radiobutton_força.grid(sticky='w',row='0',column='0', padx='10')
        self.radiobutton_rugosidade.grid(sticky='w',row='0',column='1', padx='10')
        
        self.l_tipo = ttk.Label(self,text='Tipo:')
        self.l_tipo.grid(padx='10', pady='10', row='2', sticky='e')
        
        self.t_tipo = tk.Text(self, height='1',  width='50')
        self.t_tipo.grid(column='1', padx='10', pady='10', row='2')
        
        self.l_referencia = ttk.Label(self,text='Referência:')
        self.l_referencia.grid(column='0', padx='10', pady='10', row='3', sticky='ne')
        
        self.t_referencia = tk.Text(self,height='1', undo='false', width='50')
        self.t_referencia.grid(column='1', padx='10', pady='10', row='3')
        
        self.l_material = ttk.Label(self,text='Material:')
        self.l_material.grid(column='0', padx='10', pady='10', row='4', sticky='e')
        
        self.t_material = tk.Text(self,height='1', width='50')
        self.t_material.grid(column='1', padx='10', pady='10', row='4')
        
        self.l_ferramenta = ttk.Label(self,text='Ferramenta de corte:')
        self.l_ferramenta.grid(column='0', padx='10', pady='10', row='5', sticky='e')
        
        self.t_ferramenta = tk.Text(self,height='1', state='normal', width='50')
        self.t_ferramenta.grid(column='1', padx='10', pady='10', row='5')
        
        self.l_numero = ttk.Label(self,text='N° de experimentos:')
        self.l_numero.grid(column='0', padx='10', pady='10', row='6', sticky='e')
       
        self.t_numero = tk.Text(self,height='1', width='50')
        self.t_numero.grid(column='1', padx='10', pady='10', row='6')
        
        self.l_observacoes = ttk.Label(self,text='Observações:')
        self.l_observacoes.grid(column='0', padx='10', pady='10', row='7', sticky='ne')
        
        self.t_observacoes = tk.Text(self,height='10', width='50')
        self.t_observacoes.grid(column='1', padx='10', pady='10', row='7')
        
        self.l_unidades= tk.Label(self, text= 'Unidades:')
        self.l_unidades.grid(column='2', padx='10', pady='10', row='1', sticky='nesw')
        
        self.l_u_velocidade = ttk.Label(self,text='Velocidade:')
        self.l_u_velocidade.grid(column='2', padx='10', pady='10', row='2', sticky='ne')
        
        self.t_u_velocidade = tk.Text(self,height='1', state='normal', width='15')
        self.t_u_velocidade.grid(column='3', padx='10', pady='10', row='2')
        
        self.l_u_avanco = ttk.Label(self,text='Avanço:')
        self.l_u_avanco.grid(column='2', padx='10', pady='10', row='3', sticky='ne')
        
        self.t_u_avanco = tk.Text(self,height='1', state='normal', width='15')
        self.t_u_avanco.grid(column='3', padx='10', pady='10', row='3')
        
        self.l_u_profundidade = ttk.Label(self,text='Profundidade de corte:')
        self.l_u_profundidade.grid(column='2', padx='10', pady='10', row='4', sticky='ne')
        
        self.t_u_profundidade = tk.Text(self,height='1', state='normal', width='15')
        self.t_u_profundidade.grid(column='3', padx='10', pady='10', row='4')
        
        self.l_u_saida = ttk.Label(self,text='Força / Rugosidade:')
        self.l_u_saida.grid(column='2', padx='10', pady='10', row='5', sticky='ne')
        
        self.t_u_saida = tk.Text(self,height='1', state='normal', width='15')
        self.t_u_saida.grid(column='3', padx='10', pady='10', row='5')
        
        self.b_anexar_artigo = ttk.Button(self,text='Anexar artigo',width='25')
        self.b_anexar_artigo.place(relx=0.1,rely=0.75)
        
        self.b_dados_experimentais=ttk.Button(self,text='Inserir dados experimentais',state='disabled',width='25')
        self.b_dados_experimentais.place(relx=0.35,rely=0.75)
        
        self.b_voltar = ttk.Button(self,text='Voltar')
        self.b_voltar.place(relx=0.9,rely=0.95)
        

        
        
class SampleApp2(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (popup_inserir,popup_criando,popup_pronto):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("popup_inserir")
        
    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        
class popup_inserir(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        
        self.l_validacao=ttk.Label(self,text='Validação cruzada:')
        self.l_validacao.grid(column='0',row='0',padx='10', pady='10',  sticky='e')
        
        self.e_validacao=ttk.Entry(self)
        self.e_validacao.grid(column='1',row='0')
        
        self.l_iteracoes=ttk.Label(self,text='Número de iterações:')
        self.l_iteracoes.grid(column='0',row='1',padx='10', pady='10',  sticky='e')
        
        self.e_iteracoes=ttk.Entry(self)
        self.e_iteracoes.grid(column='1',row='1')
        
        self.l_mensagem=ttk.Label(self,text='Após salvar os dados experimentais no excel aperte Ok para gerar os modelos.')
        self.l_mensagem.grid(column='0',row='2',columnspan=2,padx='10', pady='10')
        
        self.b_ok=ttk.Button(self,text='Ok')
        self.b_ok.grid(column='0',row='3',columnspan=2,padx='10', pady='10')

class popup_criando(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        
        self.l_criando=ttk.Label(self,text='Criando modelos')
        self.l_criando.place(relx=0.5, rely=0.3, anchor='center')
        self.bar=ttk.Progressbar(self, orient = 'horizontal', 
              length = 100, mode = 'indeterminate') 
        self.bar.place(relx=0.5, rely=0.7, anchor='center')
        
        
class popup_pronto(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        
        self.l_criado=ttk.Label(self,text='Modelo criado')
        self.l_criado.place(relx=0.5, rely=0.3, anchor='center')
        self.b_fechar=ttk.Button(self,text='Fechar')
        self.b_fechar.place(relx=0.5, rely=0.7, anchor='center')
        
      
    