import {cpSync, mkdirSync, rmSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {resolve} from 'node:path';

export const projectRoot=fileURLToPath(new URL('../../',import.meta.url));
export const outputDirectory=resolve(projectRoot,'dist');

export function build() {
  // Only this fixed, ignored output directory is replaced. Source stays in viz/.
  rmSync(outputDirectory,{recursive:true,force:true});
  mkdirSync(outputDirectory,{recursive:true});
  cpSync(resolve(projectRoot,'viz/src'),outputDirectory,{recursive:true});
  cpSync(resolve(projectRoot,'viz/public'),outputDirectory,{recursive:true});
  return outputDirectory;
}

if(process.argv[1] && resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  console.log(`Built ${build()}`);
}
