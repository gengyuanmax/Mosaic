const sidebar=document.querySelector('#site-sidebar');
const toggle=document.querySelector('#navigation-toggle');
const close=document.querySelector('#navigation-close');
const backdrop=document.querySelector('#navigation-backdrop');
const main=document.querySelector('main');
const mobile=matchMedia('(max-width:799px)');
const links=[...sidebar.querySelectorAll('.page-navigation a')];
const sections=links.map(link=>document.querySelector(link.getAttribute('href')));
let opened=false,frame;

function setOpen(value,{returnFocus=false}={}){
  opened=mobile.matches&&value;
  sidebar.dataset.open=String(opened);
  backdrop.dataset.open=String(opened);
  sidebar.inert=mobile.matches&&!opened;
  main.inert=opened;
  toggle.setAttribute('aria-expanded',String(opened));
  document.body.classList.toggle('navigation-open',opened);
  if(opened){sidebar.setAttribute('role','dialog');sidebar.setAttribute('aria-modal','true');close.focus();}
  else{sidebar.removeAttribute('role');sidebar.removeAttribute('aria-modal');if(returnFocus)toggle.focus();}
}
toggle.addEventListener('click',()=>setOpen(!opened));
close.addEventListener('click',()=>setOpen(false,{returnFocus:true}));
backdrop.addEventListener('click',()=>setOpen(false,{returnFocus:true}));
mobile.addEventListener('change',()=>{setOpen(false);schedule();});
document.addEventListener('keydown',event=>{
  if(!opened)return;
  if(event.key==='Escape'){event.preventDefault();setOpen(false,{returnFocus:true});}
  if(event.key==='Tab'){
    const controls=[...sidebar.querySelectorAll('a[href],button')];
    if(event.shiftKey&&document.activeElement===controls[0]){event.preventDefault();controls.at(-1).focus();}
    else if(!event.shiftKey&&document.activeElement===controls.at(-1)){event.preventDefault();controls[0].focus();}
  }
});
for(const link of sidebar.querySelectorAll('a[href^="#"]'))link.addEventListener('click',()=>{
  const wasOpen=opened;setOpen(false);
  if(wasOpen){const section=document.querySelector(link.getAttribute('href'));section.setAttribute('tabindex','-1');section.focus({preventScroll:true});}
});
function updateCurrent(){
  frame=null;let active=0;
  const threshold=mobile.matches?100:Math.min(130,innerHeight*.2);
  sections.forEach((section,index)=>{if(section.getBoundingClientRect().top<=threshold)active=index;});
  if(scrollY+innerHeight>=document.documentElement.scrollHeight-4)active=sections.length-1;
  for(const [index,link] of links.entries()){
    if(index===active)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');
    link.closest('li').classList.remove('is-ancestor');
  }
  const parent=links[active].closest('ul').closest('li');
  if(parent)parent.classList.add('is-ancestor');
}
function schedule(){if(!frame)frame=requestAnimationFrame(updateCurrent);}
window.addEventListener('scroll',schedule,{passive:true});
window.addEventListener('resize',schedule,{passive:true});
window.addEventListener('hashchange',schedule);
new ResizeObserver(schedule).observe(main);
setOpen(false);schedule();
