// Presentation labels/defaults only. Tool schemas and validation come from Python.
const labels={crop:'Crop',rotate:'Rotate',resize:'Resize',flip:'Flip',apply_affine_transformation:'Affine transform',apply_homography_transformation:'Homography',draw_normalized_coordinate_grid:'Coordinate grid',compare_per_pixel_image_difference:'Pixel difference',make_collage:'Collage',overlay_images:'Overlay'};
const fields={x1:['Left',.2,.01],y1:['Top',.2,.01],x2:['Right',.8,.01],y2:['Bottom',.8,.01],angle:['Angle (°)',30,1,-180,180],width:['Width (px)',384,1,1,2048],height:['Height (px)',384,1,1,2048],rotation_angle:['Angle (°)',20,1,-180,180],scale:['Scale',1,.05,.1,3],translation_dx:['Horizontal offset (px)',60,1,-512,512],translation_dy:['Vertical offset (px)',30,1,-512,512],grid_rows:['Rows',4,1],grid_cols:['Columns',4,1],rows:['Rows',1,1],cols:['Columns',2,1],x:['Horizontal position',.22,.01],y:['Vertical position',.22,.01]};
export function controlsFromSpecs(specs){
 return Object.keys(labels).map(id=>{
  const spec=specs.find(t=>t.name===id);if(!spec)throw Error(`Toolbox is missing ${id}.`);
  const properties=spec.arg_schema.properties;
  return {id,label:labels[id],inputs:spec.max_target_images===null?0:spec.max_target_images,
   schema:properties,params:Object.entries(properties).filter(([,p])=>['number','integer'].includes(p.type)).map(([key,p])=>{
    const display=fields[key]||[key,0,p.type==='integer'?1:.01];
    const min=p.minimum??(p.exclusiveMinimum!==undefined?p.exclusiveMinimum+(p.type==='integer'?1:.01):undefined);
    return {key,label:display[0],value:display[1],step:display[2],min,max:p.maximum,sliderMin:display[3]??min,sliderMax:display[4]??p.maximum};
   })};
 });
}
