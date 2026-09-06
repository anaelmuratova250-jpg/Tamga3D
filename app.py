<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tamga3D</title>
<style>
*{box-sizing:border-box}
body{
 margin:0;background:#111;color:white;font-family:Arial;
 text-align:center;overflow-x:hidden
}
h2{margin:12px 0}
button,input{
 margin:5px;padding:10px;border-radius:8px;border:0
}
button{background:#333;color:white}
#box{position:relative;display:inline-block;max-width:96vw}
canvas{max-width:96vw;border-radius:8px;touch-action:none}
#pick{position:absolute;left:0;top:0}
#model{
 display:block;margin:15px auto;width:96vw;height:55vh;
 background:#181818;border-radius:10px
}
.info{font-size:14px;color:#bbb}
</style>
</head>
<body>

<h2>🧶 Tamga3D</h2>
<div class="info">Выбери 4 точки по углам орнамента</div>

<input id="file" type="file" accept="image/*">

<div id="box">
 <canvas id="img"></canvas>
 <canvas id="pick"></canvas>
</div>

<br>

<button onclick="make()">Создать 3D</button>
<button onclick="reset()">Сбросить точки</button>
<button onclick="save()">Экспорт OBJ</button>

<br>

<label>Толщина
<input id="thick" type="range" min="1" max="12" value="3">
</label>

<label>Рельеф
<input id="relief" type="range" min="0" max="15" value="5">
</label>

<canvas id="model"></canvas>

<script>
const img=document.getElementById("img");
const pick=document.getElementById("pick");
const p=pick.getContext("2d");
const c=img.getContext("2d");
const model=document.getElementById("model");
const m=model.getContext("2d");

let points=[];
let image=new Image();
let vertices=[];
let faces=[];
let colors=[];

document.getElementById("file").onchange=e=>{
 const f=e.target.files[0];
 if(!f)return;
 image.onload=()=>{
  let s=Math.min(900/image.width,1);
  img.width=image.width*s;
  img.height=image.height*s;
  pick.width=img.width;
  pick.height=img.height;
  c.drawImage(image,0,0,img.width,img.height);
  points=[];
  drawPoints();
 };
 image.src=URL.createObjectURL(f);
};

function drawPoints(){
 p.clearRect(0,0,pick.width,pick.height);

 if(points.length>1){
  p.beginPath();
  p.moveTo(points[0].x,points[0].y);
  for(let i=1;i<points.length;i++)
   p.lineTo(points[i].x,points[i].y);
  if(points.length==4)p.closePath();
  p.strokeStyle="#00ffff";
  p.lineWidth=3;
  p.stroke();
 }

 points.forEach((q,i)=>{
  p.beginPath();
  p.arc(q.x,q.y,10,0,Math.PI*2);
  p.fillStyle="#ffcc00";
  p.fill();
  p.strokeStyle="white";
  p.lineWidth=2;
  p.stroke();

  p.fillStyle="black";
  p.font="bold 12px Arial";
  p.fillText(i+1,q.x-4,q.y+4);
 });
}

let drag=-1;

pick.addEventListener("pointerdown",e=>{
 const r=pick.getBoundingClientRect();
 const x=(e.clientX-r.left)*pick.width/r.width;
 const y=(e.clientY-r.top)*pick.height/r.height;

 drag=-1;

 for(let i=0;i<points.length;i++){
  if(Math.hypot(points[i].x-x,points[i].y-y)<25){
   drag=i;
   pick.setPointerCapture(e.pointerId);
   return;
  }
 }

 if(points.length<4){
  points.push({x,y});
  drawPoints();
 }
});

pick.addEventListener("pointermove",e=>{
 if(drag<0)return;

 const r=pick.getBoundingClientRect();
 points[drag].x=(e.clientX-r.left)*pick.width/r.width;
 points[drag].y=(e.clientY-r.top)*pick.height/r.height;

 drawPoints();
});

pick.addEventListener("pointerup",()=>{
 drag=-1;
});

function reset(){
 points=[];
 drawPoints();
}

function make(){
 if(points.length!==4){
  alert("Сначала поставь 4 точки");
  return;
 }

 vertices=[];
 faces=[];
 colors=[];

 let W=45,H=45;
 let w=model.width=700;
 let h=model.height=500;

 /*
   Четыре выбранные точки:
   0 = левый верх
   1 = правый верх
   2 = правый низ
   3 = левый низ
 */

 let src=[
  points[0],points[1],points[2],points[3]
 ];

 let dst=[
  {x:0,y:0},
  {x:W,y:0},
  {x:W,y:H},
  {x:0,y:H}
 ];

 let map=homography(src,dst);

 let data=c.getImageData(0,0,img.width,img.height);

 function sample(x,y){
  let X=map[0]*x+map[1]*y+map[2];
  let Y=map[3]*x+map[4]*y+map[5];
  let Z=map[6]*x+map[7]*y+1;

  X/=Z;
  Y/=Z;

  X=Math.max(0,Math.min(img.width-1,X));
  Y=Math.max(0,Math.min(img.height-1,Y));

  let k=(Math.floor(Y)*img.width+Math.floor(X))*4;

  return [
   data.data[k],
   data.data[k+1],
   data.data[k+2]
  ];
 }

 let t=+document.getElementById("thick").value;
 let rel=+document.getElementById("relief").value;

 for(let y=0;y<=H;y++){
  for(let x=0;x<=W;x++){

   let uv=sample(x,y);
   let bright=(uv[0]+uv[1]+uv[2])/765;

   let z=bright*rel;

   vertices.push([
    x-W/2,
    H/2-y,
    z
   ]);

   colors.push(uv);
 }
 }

 for(let y=0;y<H;y++){
  for(let x=0;x<W;x++){
   let a=y*(W+1)+x;
   let b=a+1;
   let d=(y+1)*(W+1)+x;
   let e=d+1;

   faces.push([a,b,e,d]);
  }
 }

 /*
   Рисуем 3D-рельеф.
   Цвет каждой точки берётся прямо
   из исходного изображения.
 */

 draw3D(W,H,t);
}

function draw3D(W,H,t){

 m.clearRect(0,0,model.width,model.height);

 let rotX=-.45;
 let rotY=.6;

 function project(v){
  let x=v[0],y=v[1],z=v[2];

  let cy=Math.cos(rotY),sy=Math.sin(rotY);
  let x1=x*cy-z*sy;
  let z1=x*sy+z*cy;

  let cx=Math.cos(rotX),sx=Math.sin(rotX);
  let y1=y*cx-z1*sx;
  let z2=y*sx+z1*cx;

  let scale=8/(8+z2/80);

  return [
   model.width/2+x1*scale,
   model.height/2-y1*scale
  ];
 }

 faces.forEach(f=>{
  let q=f.map(i=>project(vertices[i]));

  let col=colors[f[0]];
  m.beginPath();
  m.moveTo(q[0][0],q[0][1]);

  for(let i=1;i<q.length;i++)
   m.lineTo(q[i][0],q[i][1]);

  m.closePath();

  m.fillStyle=`rgb(${col[0]},${col[1]},${col[2]})`;
  m.fill();
 });

 /*
   Шерстяной эффект:
   короткие волокна поверх цветной поверхности
 */

 for(let i=0;i<vertices.length;i+=2){

  let v=vertices[i];
  let q=project(v);
  let col=colors[i];

  let len=2+v[2]/4;

  m.strokeStyle=
   `rgb(${Math.min(255,col[0]+25)},
        ${Math.min(255,col[1]+25)},
        ${Math.min(255,col[2]+25)})`;

  m.lineWidth=1;

  m.beginPath();
  m.moveTo(q[0],q[1]);
  m.lineTo(q[0]+Math.sin(i)*len,
           q[1]-Math.cos(i)*len);
  m.stroke();
 }
}

/*
  Гомография.
  Она превращает любой четырёхугольник
  в ровный прямоугольник.
 */

function homography(s,d){

 let A=[],B=[];

 for(let i=0;i<4;i++){
  let x=s[i].x,y=s[i].y;
  let u=d[i].x,v=d[i].y;

  A.push([x,y,1,0,0,0,-u*x,-u*y]);
  B.push(u);

  A.push([0,0,0,x,y,1,-v*x,-v*y]);
  B.push(v);
 }

 for(let i=0;i<8;i++){
  let max=i;

  for(let j=i+1;j<8;j++)
   if(Math.abs(A[j][i])>Math.abs(A[max][i]))
    max=j;

  [A[i],A[max]]=[A[max],A[i]];
  [B[i],B[max]]=[B[max],B[i]];

  let z=A[i][i];

  for(let k=i;k<8;k++)A[i][k]/=z;
  B[i]/=z;

  for(let j=0;j<8;j++){
   if(j==i)continue;

   let q=A[j][i];

   for(let k=i;k<8;k++)
    A[j][k]-=q*A[i][k];

   B[j]-=q*B[i];
  }
 }

 return [
  B[0],B[1],B[2],
  B[3],B[4],B[5],
  B[6],B[7]
 ];
}

function save(){

 if(!vertices.length){
  alert("Сначала создай 3D");
  return;
 }

 let out="# Tamga3D\n";

 vertices.forEach(v=>{
  out+=`v ${v[0]} ${v[1]} ${v[2]}\n`;
 });

 faces.forEach(f=>{
  out+=`f ${f[0]+1} ${f[1]+1} ${f[2]+1} ${f[3]+1}\n`;
 });

 let blob=new Blob([out],{type:"text/plain"});
 let a=document.createElement("a");
 a.href=URL.createObjectURL(blob);
 a.download="tamga3d.obj";
 a.click();
}
</script>
</body>
</html>
