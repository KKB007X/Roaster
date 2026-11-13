/* Standalone test of slot_save_dir / slot_restore_dir (no weights, no GPU needed; kernels and the
 * tokenizer are stubbed out by the linker):
 *   gcc -g -fsanitize=address,undefined -fopenmp -Wl,--unresolved-symbols=ignore-all -I. \
 *       test_slot_save.c slots.c model.c state.c kvwindow.c kvspill.c -o test_slot_save -lm -lpthread
 *   ASAN_OPTIONS=detect_leaks=0 ./test_slot_save
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "slots.h"
#include "kvwindow.h"
TensorInfo *weights_find_tensor(Weights *w, const char *n){ (void)w;(void)n; return NULL; }
static int fails;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %d: %s\n",__LINE__,#c); fails++; } }while(0)
static void fill(Model*m){ for(int l=0;l<MODEL_NUM_LAYERS;l++) if(model_is_attention_layer(l)){ KVCache*kv=&m->kv_cache[l]; for(int j=0;j<512;j++){kv->k[kv->length*512+j]=(float)m->position+100000.0f*l; kv->v[kv->length*512+j]=-(float)m->position-l;} kv->length++; } m->position++; }
static void add(Model*m,int n){ for(int i=0;i<n;i++){ if(kvw_ensure_room(m,1)){puts("room fail");exit(1);} fill(m);} }
/* every window entry's value equals an absolute position: collect them */
static size_t positions(Model*m,size_t*out){ KVCache*kv=&m->kv_cache[3]; for(size_t i=0;i<kv->length;i++) out[i]=(size_t)(kv->k[i*512]-300000.0f); return kv->length; }
static int same(Model*m,size_t*a,size_t n){ size_t b[5000]; size_t k=positions(m,b); if(k!=n) return 0; for(size_t i=0;i<n;i++) if(a[i]!=b[i]) return 0; 
  for(int l=0;l<MODEL_NUM_LAYERS;l++) if(model_is_attention_layer(l)){ KVCache*kv=&m->kv_cache[l]; for(size_t i=0;i<n;i++) if(kv->k[i*512+7]!=(float)a[i]+100000.0f*l||kv->v[i*512+7]!=-(float)a[i]-l) return 0;} return 1; }
int main(void){
  system("rm -rf /tmp/rt_state /tmp/rt_root /tmp/rt_ctx"); mkdir("/tmp/rt_state",0700);
  Weights w; memset(&w,0,sizeof w); w.fd=open("/etc/hostname",O_RDONLY); w.tensor_count=7;
  SlotManager sm; memset(&sm,0,sizeof sm); sm.slots=calloc(2,sizeof(InferenceSlot)); sm.weights=&w; sm.max_slots=2;
  snprintf(sm.state_dir,sizeof sm.state_dir,"/tmp/rt_state"); pthread_mutex_init(&sm.table_lock,NULL);
  SlotConfig c; slot_config_defaults(&c); c.context_length=64; c.keep=4;
  int id=slot_create(&sm,&c); CHECK(id>=0);
  Model*m=&sm.slots[id].model; char err[512]; size_t bytes;
  CHECK(slot_set_system(&sm,id,"be nice\nline2",err,sizeof err)==0);
  m->slide_pin=10; sm.slots[id].pinned=10;
  add(m,500);
  size_t before[5000]; size_t n=positions(m,before); size_t pos=m->position, sp=kvw_spilled(m);
  CHECK(sp>0);
  CHECK(slot_save_dir(&sm,id,"/tmp/rt_ctx/a/b",&bytes,err,sizeof err)==0 && bytes>0);
  /* wipe context, restore */
  CHECK(slot_reset(&sm,id,err,sizeof err)==0 && m->position==0 && kvw_spilled(m)==0);
  CHECK(slot_restore_dir(&sm,id,"/tmp/rt_ctx/a/b",err,sizeof err)==0);
  if(err[0]&&0) puts(err);
  CHECK(m->position==pos && kvw_spilled(m)==sp && same(m,before,n) && m->slide_pin==10 && sm.slots[id].pinned==10);
  CHECK(sm.slots[id].system && strcmp(sm.slots[id].system,"be nice\nline2")==0);
  add(m,100); { size_t b2[5000]; size_t k=positions(m,b2); CHECK(k<=64); }  /* keeps sliding afterwards */
  /* restore into a bigger window: spill is brought back */
  CHECK(slot_restore_dir(&sm,id,"/tmp/rt_ctx/a/b",err,sizeof err)==0);
  { SlotConfig nc=sm.slots[id].config; nc.context_length=4096; CHECK(slot_configure(&sm,id,&nc,err,sizeof err)==0); }
  CHECK(kvw_spilled(m)==0 && kvw_len(m)==m->position);
  /* restore the 64-window snapshot while the slot has 4096: window stays 4096, all comes back */
  CHECK(slot_restore_dir(&sm,id,"/tmp/rt_ctx/a/b",err,sizeof err)==0);
  CHECK(sm.slots[id].config.context_length==4096 && kvw_spilled(m)==0 && kvw_len(m)==pos);
  /* smaller current window grows to the saved size */
  { SlotConfig nc=sm.slots[id].config; nc.context_length=32; CHECK(slot_configure(&sm,id,&nc,err,sizeof err)==0); }
  CHECK(slot_restore_dir(&sm,id,"/tmp/rt_ctx/a/b",err,sizeof err)==0);
  CHECK(sm.slots[id].config.context_length==64 && m->position==pos);
  /* errors */
  CHECK(slot_restore_dir(&sm,id,"/tmp/rt_ctx/nope",err,sizeof err)!=0 && strstr(err,"no saved context"));
  CHECK(slot_restore_dir(&sm,id,"/tmp/../etc",err,sizeof err)!=0 && strstr(err,"invalid path"));
  CHECK(slot_save_dir(&sm,id,"",&bytes,err,sizeof err)!=0 && strstr(err,"invalid path"));
  /* corrupt metadata */
  { FILE*f=fopen("/tmp/rt_ctx/a/b/roaster-context.meta","a"); fclose(f); system("sed -i 's/^model=.*/model=1-1/' /tmp/rt_ctx/a/b/roaster-context.meta");
    CHECK(slot_restore_dir(&sm,id,"/tmp/rt_ctx/a/b",err,sizeof err)!=0 && strstr(err,"different model")); }
  /* unloaded slot can be saved */
  CHECK(slot_save_dir(&sm,id,"/tmp/rt_ctx/c",&bytes,err,sizeof err)==0);
  CHECK(slot_unload(&sm,id,0,err,sizeof err)==0);
  CHECK(slot_save_dir(&sm,id,"/tmp/rt_ctx/d",&bytes,err,sizeof err)==0 && bytes>0);
  CHECK(slot_restore_dir(&sm,id,"/tmp/rt_ctx/c",err,sizeof err)!=0 && strstr(err,"unloaded"));
  CHECK(slot_load(&sm,id,err,sizeof err)==0);
  CHECK(slot_reset(&sm,id,err,sizeof err)==0);
  CHECK(slot_restore_dir(&sm,id,"/tmp/rt_ctx/d",err,sizeof err)==0 && sm.slots[id].model.position==pos);
  /* save root */
  slots_set_save_root(&sm,"/tmp/rt_root");
  CHECK(slot_save_dir(&sm,id,"/abs/x",&bytes,err,sizeof err)==0 && access("/tmp/rt_root/abs/x/roaster-context.meta",0)==0);
  CHECK(slot_restore_dir(&sm,id,"abs/x",err,sizeof err)==0);
  slot_destroy(&sm,id);
  printf(fails?"FAILED\n":"SLOT SAVE OK\n"); return fails!=0; }
