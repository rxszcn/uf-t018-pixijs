import { ExtensionType } from '../../../extensions/Extensions';
import { CLEAR } from '../../renderers/gl/const';
import { STENCIL_MODES } from '../../renderers/shared/state/const';

import type { Container } from '../../../scene/container/Container';
import type { Effect } from '../../../scene/container/Effect';
import type { WebGLRenderer } from '../../renderers/gl/WebGLRenderer';
import type { WebGPURenderer } from '../../renderers/gpu/WebGPURenderer';
import type { Instruction } from '../../renderers/shared/instructions/Instruction';
import type { InstructionSet } from '../../renderers/shared/instructions/InstructionSet';
import type { InstructionPipe } from '../../renderers/shared/instructions/RenderPipe';
import type { Renderable } from '../../renderers/shared/Renderable';
import type { Renderer } from '../../renderers/types';
import type { StencilMask } from './StencilMask';
import type { StencilMaskInstruction } from './StencilMaskTypes';

/**
 * The instructions collected for a mask, recorded so they can be replayed when the
 * mask is popped or when another container shares the same mask.
 * @internal
 */
interface MaskInstructionBlock
{
    /** the instruction set the block was collected into */
    instructionSet: InstructionSet;
    /** the index of the first instruction of the block */
    instructionsStart: number;
    /** how many instructions the block contains */
    instructionsLength: number;
    /** the instruction terminating the block, used to detect blocks left over from a previous build */
    blockEnd: Instruction;
}

/** @internal */
export class StencilMaskPipe implements InstructionPipe<StencilMaskInstruction>
{
    public static extension = {
        type: [
            ExtensionType.WebGLPipes,
            ExtensionType.WebGPUPipes,
        ],
        name: 'stencilMask',
    } as const;

    private _renderer: Renderer;

    private _maskHash = new WeakMap<Container, MaskInstructionBlock>();

    constructor(renderer: Renderer)
    {
        this._renderer = renderer;
    }

    public push(mask: Effect, _container: Container, instructionSet: InstructionSet): void
    {
        const effect = mask as StencilMask;

        const renderer = this._renderer;

        renderer.renderPipes.batch.break(instructionSet);

        renderer.renderPipes.blendMode.setBlendMode(effect.mask as Renderable, 'none', instructionSet);

        instructionSet.add({
            renderPipeId: 'stencilMask',
            action: 'pushMaskBegin',
            mask,
            inverse: _container._maskOptions.inverse,
            canBundle: false,
        } as StencilMaskInstruction);

        const maskContainer = effect.mask;

        const maskData = this._maskHash.get(maskContainer);

        if (this._isBlockValid(maskData, instructionSet))
        {
            // the mask is shared with another container and its instructions were already
            // collected into this instruction set - replay them instead of collecting again.
            // this keeps a single copy of the mask geometry in the batcher, so when the mask
            // moves, its vertices are updated in place for every container that shares it
            this._copyMaskInstructions(maskData, instructionSet);

            instructionSet.add({
                renderPipeId: 'stencilMask',
                action: 'pushMaskEnd',
                mask,
                inverse: _container._maskOptions.inverse,
                canBundle: false,
            } as StencilMaskInstruction);

            return;
        }

        maskContainer.includeInBuild = true;

        const instructionsStart = instructionSet.instructionSize;

        maskContainer.collectRenderables(
            instructionSet,
            renderer,
            null
        );

        maskContainer.includeInBuild = false;

        renderer.renderPipes.batch.break(instructionSet);

        const instructionsLength = instructionSet.instructionSize - instructionsStart;

        const blockEnd = {
            renderPipeId: 'stencilMask',
            action: 'pushMaskEnd',
            mask,
            inverse: _container._maskOptions.inverse,
            canBundle: false,
        } as StencilMaskInstruction;

        instructionSet.add(blockEnd);

        this._maskHash.set(maskContainer, {
            instructionSet,
            instructionsStart,
            instructionsLength,
            blockEnd,
        });
    }

    public pop(mask: Effect, _container: Container, instructionSet: InstructionSet): void
    {
        const effect = mask as StencilMask;

        const renderer = this._renderer;

        // stencil is stored based on current render target..
        renderer.renderPipes.batch.break(instructionSet);
        renderer.renderPipes.blendMode.setBlendMode(effect.mask as Renderable, 'none', instructionSet);

        instructionSet.add({
            renderPipeId: 'stencilMask',
            action: 'popMaskBegin',
            inverse: _container._maskOptions.inverse,
            canBundle: false,
        } as StencilMaskInstruction);

        const maskData = this._maskHash.get(effect.mask);

        this._copyMaskInstructions(maskData, instructionSet);

        instructionSet.add({
            renderPipeId: 'stencilMask',
            action: 'popMaskEnd',
            canBundle: false,
        });
    }

    private _isBlockValid(maskData: MaskInstructionBlock, instructionSet: InstructionSet): boolean
    {
        if (!maskData || maskData.instructionSet !== instructionSet) return false;

        const blockEndIndex = maskData.instructionsStart + maskData.instructionsLength;

        // the block only belongs to the current build of the instruction set if it lies
        // behind the write head and its terminating instruction is still intact - an
        // instruction set is reused across builds, so positions alone cannot be trusted
        return instructionSet.instructionSize > blockEndIndex
            && instructionSet.instructions[blockEndIndex] === maskData.blockEnd;
    }

    private _copyMaskInstructions(maskData: MaskInstructionBlock, instructionSet: InstructionSet): void
    {
        const instructions = instructionSet.instructions;
        const instructionsStart = maskData.instructionsStart;
        const instructionsLength = maskData.instructionsLength;

        for (let i = 0; i < instructionsLength; i++)
        {
            instructions[instructionSet.instructionSize++] = instructions[instructionsStart + i];
        }
    }

    public execute(instruction: StencilMaskInstruction)
    {
        const renderer = this._renderer;

        const gpuRenderer = renderer as WebGLRenderer | WebGPURenderer;
        const gpuRenderTarget = gpuRenderer.renderTarget.getGpuRenderTarget(gpuRenderer.renderTarget.renderTarget);

        let maskStackIndex = gpuRenderTarget.maskStackIndex;

        if (instruction.action === 'pushMaskBegin')
        {
            // we create the depth and stencil buffers JIT
            // as no point allocating the memory if we don't use it
            gpuRenderer.renderTarget.ensureDepthStencil();

            gpuRenderer.stencil.setStencilMode(STENCIL_MODES.RENDERING_MASK_ADD, maskStackIndex);

            maskStackIndex++;

            gpuRenderer.colorMask.setMask(0);
        }
        else if (instruction.action === 'pushMaskEnd')
        {
            if (instruction.inverse)
            {
                gpuRenderer.stencil.setStencilMode(STENCIL_MODES.INVERSE_MASK_ACTIVE, maskStackIndex);
            }
            else
            {
                gpuRenderer.stencil.setStencilMode(STENCIL_MODES.MASK_ACTIVE, maskStackIndex);
            }

            gpuRenderer.colorMask.setMask(0xF);
        }
        else if (instruction.action === 'popMaskBegin')
        {
            gpuRenderer.colorMask.setMask(0);

            if (maskStackIndex !== 0)
            {
                gpuRenderer.stencil.setStencilMode(STENCIL_MODES.RENDERING_MASK_REMOVE, maskStackIndex);
            }
            else
            {
                gpuRenderer.renderTarget.clear(null, CLEAR.STENCIL);
                gpuRenderer.stencil.setStencilMode(STENCIL_MODES.DISABLED, maskStackIndex);
            }

            maskStackIndex--;
        }
        else if (instruction.action === 'popMaskEnd')
        {
            if (instruction.inverse)
            {
                gpuRenderer.stencil.setStencilMode(STENCIL_MODES.INVERSE_MASK_ACTIVE, maskStackIndex);
            }
            else
            {
                gpuRenderer.stencil.setStencilMode(STENCIL_MODES.MASK_ACTIVE, maskStackIndex);
            }

            gpuRenderer.colorMask.setMask(0xF);
        }

        gpuRenderTarget.maskStackIndex = maskStackIndex;
    }

    public destroy()
    {
        this._renderer = null;
        this._maskHash = null;
    }
}
